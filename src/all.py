from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DemonstrationDataset
from policy_network import PolicyNetwork
from torch.optim.lr_scheduler import CosineAnnealingLR
from onnx2pytorch import ConvertModel
from env_utils import Agent, make_env
import os
import time
import onnx
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    n_epochs = 30
    train_set = DemonstrationDataset("data/train", augment=True)
    val_set = DemonstrationDataset("data/val", augment=False)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
    model = PolicyNetwork(num_actions=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_entropy_sum = 0.0
        num_samples = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Training]", leave=False, dynamic_ncols=True)
        for states, actions in train_bar:
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
            batch_size_current = states.size(0)
            train_loss_sum += loss.item() * batch_size_current
            num_samples += batch_size_current
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=logits.float())
                entropy = dist.entropy().mean()
            train_entropy_sum += entropy.item() * batch_size_current
            train_bar.set_postfix({
                "Batch Loss": f"{loss.item():.3f}",
                "Avg Loss": f"{train_loss_sum/num_samples:.3f}",
                "Entropy": f"{entropy.item():.3f}"
            })
        train_bar.close()
        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        avg_train_entropy = train_entropy_sum / len(train_loader.dataset)
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Validation]", leave=False, dynamic_ncols=True)
        for states, actions in val_bar:
            states, actions = states.to(device), actions.to(device)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(states)
                loss = criterion(logits, actions)
            val_loss_sum += loss.item() * states.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == actions).sum().item()
            total += states.size(0)
            val_bar.set_postfix({"Batch Loss": f"{loss.item():.3f}"})
        val_bar.close()
        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc = correct / total
        epoch_time = time.time() - epoch_start
        tqdm.write(
            f"\nEpoch {epoch+1}/{n_epochs} Summary: Time: {epoch_time:.1f}s | "
            f"Train Loss: {avg_train_loss:.3f} (Entropy: {avg_train_entropy:.3f}) | "
            f"Val Loss: {avg_val_loss:.3f} (Acc: {val_acc:.3f})\n"
        )
    os.makedirs("models", exist_ok=True)
    sample_state = torch.rand(1, 1, 84, 84, device=device)
    torch.onnx.export(model, sample_state, "models/bc_model.onnx", export_params=True, opset_version=17, do_constant_folding=True)
    torch.save(model.state_dict(), os.path.join("models", "bc_model.pth"))

    bc_folder = "data/train"
    dagger_folder = "data/train_dagger"
    if not os.path.exists(dagger_folder):
        os.makedirs(dagger_folder)
        bc_files = [file for file in os.listdir(bc_folder) if file.endswith(".npz")]
        for file in tqdm(bc_files, desc="Copying BC training data", leave=False):
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
    dagger_dataset = DemonstrationDataset(dagger_folder, augment=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    n_dagger_iters = 15
    dagger_episodes_per_iter = 10
    training_epochs_per_iter = 10
    best_avg_reward = -float('inf')
    best_model_state = None
    for i in tqdm(range(n_dagger_iters), desc="DAgger Iterations", leave=False, dynamic_ncols=True):
        iter_start = time.time()
        env = make_env(seed=123 + i)
        beta = 1.0 - (i / (n_dagger_iters - 1)) * 0.5
        episode_rewards = []
        for ep in tqdm(range(dagger_episodes_per_iter), desc=f"Iter {i+1} Episodes", leave=False, dynamic_ncols=True):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0
            while not done:
                full_obs = np.array(obs)
                student_input = np.expand_dims(full_obs[-1], 0)
                expert_input = full_obs
                if np.random.rand() < beta:
                    action = expert_agent.select_action(expert_input)
                else:
                    action = train_agent.select_action(student_input)
                expert_action_val = expert_agent.select_action(expert_input)
                expert_action_np = np.array(expert_action_val)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step_count += 1
                dagger_dataset.append([full_obs[-1]], [expert_action_np])
            episode_rewards.append(ep_reward)
            tqdm.write(f"Iter {i+1} Ep {ep+1}: Steps={step_count}, Reward={ep_reward:.2f}")
        avg_episode_reward = np.mean(episode_rewards)
        dagger_loader = DataLoader(dagger_dataset, batch_size=64, shuffle=True, num_workers=4)
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
            avg_epoch_loss = epoch_loss_sum / len(dagger_loader.dataset)
            tqdm.write(f"DAgger Iter {i+1} Training Epoch {epoch+1}: Loss = {avg_epoch_loss:.3f}")
            total_epoch_loss += epoch_loss_sum
        avg_loss = total_epoch_loss / (len(dagger_loader.dataset) * training_epochs_per_iter)
        iter_time = time.time() - iter_start
        tqdm.write(
            f"[DAgger Iter {i+1}/{n_dagger_iters}] Beta={beta:.2f} | Loss: {avg_loss:.3f} | "
            f"Time: {iter_time:.1f}s | Avg Ep Reward: {avg_episode_reward:.2f}"
        )
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