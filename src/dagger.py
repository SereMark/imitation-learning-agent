import os
import time
import onnx
import torch
import shutil
import numpy as np
import gymnasium as gym
import torch.optim as optim
from tqdm import tqdm
from gymnasium.spaces import Box
from onnx2pytorch import ConvertModel
from torch.utils.data import DataLoader
from dataset import DemonstrationDataset
from policy_network import PolicyNetwork
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from gymnasium.wrappers import RecordEpisodeStatistics, ResizeObservation, GrayScaleObservation, FrameStack

class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        return observation[:self.shape[0], :self.shape[1]]

class Agent():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def select_action(self, state, expert=False):
        with torch.no_grad():
            state = np.array(state)
            if expert:
                state_tensor = torch.Tensor(state).unsqueeze(0).to(self.device) / 255.0
            else:
                if state.ndim == 3:
                    state = state[-1]
                state_tensor = torch.Tensor(state).unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            logits = self.model(state_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = Categorical(logits=logits)
            return probs.sample().cpu().numpy()[0]

class RecordState(gym.Wrapper):
    def __init__(self, env, reset_clean=True):
        super().__init__(env)
        self.frame_list = []
        self.reset_clean = reset_clean

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.frame_list.append(obs)
        return obs, rew, term, trunc, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self.reset_clean:
            self.frame_list = []
        self.frame_list.append(obs)
        return obs, info

    def render(self):
        frames = self.frame_list
        self.frame_list = []
        return frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    bc_folder = "data/train"
    dagger_folder = "data/train_dagger"
    if not os.path.exists(dagger_folder):
        os.makedirs(dagger_folder)
        for file in [f for f in os.listdir(bc_folder) if f.endswith(".npz")]:
            shutil.copyfile(os.path.join(bc_folder, file), os.path.join(dagger_folder, file))

    expert_onnx = onnx.load("data/expert.onnx")
    expert_model = ConvertModel(expert_onnx).to(device)
    for p in expert_model.parameters():
        p.requires_grad = False
    expert_agent = Agent(expert_model, device)

    bc_state_dict = torch.load("models/bc_model.pth", map_location=device)
    model = PolicyNetwork(num_actions=5, in_channels=1).to(device)
    model.load_state_dict(bc_state_dict)
    train_agent = Agent(model, device)

    dagger_dataset = DemonstrationDataset(dagger_folder, augment=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    n_dagger_iters = 20
    dagger_episodes_per_iter = 20
    training_epochs_per_iter = 10
    best_avg_reward = -float("inf")
    best_model_state = None
    final_beta = 0.1

    for i in tqdm(range(n_dagger_iters), desc="DAgger Iterations", leave=False, dynamic_ncols=True):
        iter_start = time.time()
        env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
        env = RecordEpisodeStatistics(env)
        env = CropObservation(env, (84, 96))
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env)
        env = RecordState(env, reset_clean=True)
        env = FrameStack(env, 4)
        beta = np.exp((np.log(final_beta) / (n_dagger_iters - 1)) * i)
        episode_rewards = []

        for ep in tqdm(range(dagger_episodes_per_iter), desc=f"Iter {i+1} Episodes", leave=False, dynamic_ncols=True):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0

            while not done:
                expert_action = expert_agent.select_action(obs, expert=True)
                train_action = train_agent.select_action(obs, expert=False)
                action = expert_action if np.random.rand() < beta else train_action
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step_count += 1
                dagger_dataset.append([np.expand_dims(obs[-1], 0)], [expert_action])
            episode_rewards.append(ep_reward)
            tqdm.write(f"Iter {i+1} Ep {ep+1}: Steps={step_count}, Reward={ep_reward:.2f}")

        avg_episode_reward = np.mean(episode_rewards)
        dagger_loader = DataLoader(dagger_dataset, batch_size=64, shuffle=True, num_workers=4)
        dagger_scheduler = CosineAnnealingLR(optimizer, T_max=training_epochs_per_iter)
        total_epoch_loss = 0.0

        for epoch in range(training_epochs_per_iter):
            model.train()
            epoch_loss_sum = 0.0
            for states, actions in tqdm(dagger_loader, desc=f"Iter {i+1} Training Epoch {epoch+1}", leave=False, dynamic_ncols=True):
                states, actions = states.to(device), actions.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    logits = model(states)
                    loss = criterion(logits, actions)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                epoch_loss_sum += loss.item() * states.size(0)
            dagger_scheduler.step()
            avg_epoch_loss = epoch_loss_sum / len(dagger_loader.dataset)
            tqdm.write(f"DAgger Iter {i+1} Training Epoch {epoch+1}: Loss = {avg_epoch_loss:.3f}")
            total_epoch_loss += epoch_loss_sum

        avg_loss = total_epoch_loss / (len(dagger_loader.dataset) * training_epochs_per_iter)
        iter_time = time.time() - iter_start
        tqdm.write(f"[DAgger Iter {i+1}/{n_dagger_iters}] Beta={beta:.3f} | Loss: {avg_loss:.3f} | Time: {iter_time:.1f}s | Avg Ep Reward: {avg_episode_reward:.2f}")
        if avg_episode_reward > best_avg_reward:
            best_avg_reward = avg_episode_reward
            best_model_state = model.state_dict().copy()
            tqdm.write(f"New best model at iteration {i+1} with Avg Ep Reward: {best_avg_reward:.2f}")
        train_agent = Agent(model, device)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    os.makedirs("models", exist_ok=True)
    sample_state = torch.rand(1, 1, 84, 84, device=device)
    torch.onnx.export(model, sample_state, "models/dagger_model.onnx", export_params=True, opset_version=17, do_constant_folding=True)
    torch.save(model.state_dict(), os.path.join("models", "dagger_model.pth"))