from tqdm import tqdm
from onnx2pytorch import ConvertModel
from torch.utils.data import DataLoader
from src.env_utils import Agent, make_env
from src.dataset import DemonstrationDataset
from src.policy_network import PolicyNetwork
import os, time, onnx, torch, shutil, numpy as np, torch.nn as nn, torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_with_dagger():
    bc_folder = "data/train"
    dagger_folder = "data/train_dagger"

    if not os.path.exists(dagger_folder):
        os.makedirs(dagger_folder)
        bc_files = [file for file in os.listdir(bc_folder) if file.endswith(".npz")]
        for file in tqdm(bc_files, desc="Copying training data", leave=False):
            shutil.copyfile(os.path.join(bc_folder, file), os.path.join(dagger_folder, file))

    expert_onnx = onnx.load("data/expert.onnx")
    expert_model = ConvertModel(expert_onnx).to(device)
    for p in expert_model.parameters():
        p.requires_grad = False
    expert_agent = Agent(expert_model, device)

    bc_state_dict = torch.load("models/bc_model.pth", map_location=device)
    model = PolicyNetwork(num_actions=5).to(device)
    model.load_state_dict(bc_state_dict)
    train_agent = Agent(model, device)

    dagger_dataset = DemonstrationDataset(dagger_folder)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_dagger_iters = 5
    dagger_episodes_per_iter = 2
    training_epochs_per_iter = 1

    for i in tqdm(range(n_dagger_iters), desc="DAgger Iterations", leave=False):
        iter_start = time.time()
        env = make_env(seed=123 + i, capture_video=False)
        beta = 1.0 - (i / (n_dagger_iters - 1)) * 0.8 if n_dagger_iters > 1 else 1.0

        episode_rewards = []
        for ep in tqdm(range(dagger_episodes_per_iter), desc=f"Iter {i+1} Episodes", leave=False):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0
            while not done:
                current_state = np.array(obs)
                if np.random.rand() < beta:
                    action = expert_agent.select_action(current_state)
                else:
                    action = train_agent.select_action(np.expand_dims(current_state[-1], 0))

                expert_action_val = expert_agent.select_action(current_state)
                expert_action_np = np.array(expert_action_val)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step_count += 1

                dagger_dataset.append([current_state[-1]], [expert_action_np])
            episode_rewards.append(ep_reward)
            tqdm.write(f"Iter {i+1} Ep {ep+1}: Steps={step_count}, Reward={ep_reward:.2f}")

        dagger_loader = DataLoader(dagger_dataset, batch_size=64, shuffle=True)
        total_epoch_loss = 0.0
        for epoch in range(training_epochs_per_iter):
            model.train()
            epoch_loss_sum = 0.0
            train_bar = tqdm(dagger_loader, desc=f"Iter {i+1} Training", leave=False)
            for states, actions in train_bar:
                states, actions = states.to(device), actions.to(device)
                logits = model(states)
                loss = criterion(logits, actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item() * states.size(0)
                train_bar.set_postfix({"Batch Loss": f"{loss.item():.3f}"})
            avg_epoch_loss = epoch_loss_sum / len(dagger_loader.dataset)
            print(f"DAgger Iter {i+1} Training Epoch {epoch+1}: Loss = {avg_epoch_loss:.3f}")
            total_epoch_loss += epoch_loss_sum
        avg_loss = total_epoch_loss / (len(dagger_loader.dataset) * training_epochs_per_iter)
        iter_time = time.time() - iter_start

        print(
            f"[DAgger Iter {i+1}/{n_dagger_iters}] "
            f"Beta={beta:.2f} | Loss: {avg_loss:.3f} | Time: {iter_time:.1f}s | "
            f"Avg Ep Reward: {np.mean(episode_rewards):.2f}"
        )
        train_agent = Agent(model, device)

    os.makedirs("models", exist_ok=True)
    sample_state = torch.rand(1, 1, 84, 84, device=device)
    torch.onnx.export(model, sample_state, "models/dagger_model.onnx", export_params=True, opset_version=17, do_constant_folding=True)

if __name__ == "__main__":
    train_with_dagger()