import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DemonstrationDataset
from policy_network import PolicyNetwork
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    n_epochs = 50  
    train_set = DemonstrationDataset("data/train", augment=True)
    val_set = DemonstrationDataset("data/val", augment=False)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
    
    model = PolicyNetwork(num_actions=5, in_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

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

            bs = states.size(0)
            train_loss_sum += loss.item() * bs
            num_samples += bs

            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=logits.float())
                entropy = dist.entropy().mean()
            train_entropy_sum += entropy.item() * bs

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
        tqdm.write(f"\nEpoch {epoch+1}/{n_epochs} Summary: Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.3f} (Entropy: {avg_train_entropy:.3f}) | Val Loss: {avg_val_loss:.3f} (Acc: {val_acc:.3f})\n")

    os.makedirs("models", exist_ok=True)
    sample_state = torch.rand(1, 1, 84, 84, device=device)
    torch.onnx.export(model, sample_state, "models/bc_model.onnx", export_params=True, opset_version=17, do_constant_folding=True)
    torch.save(model.state_dict(), os.path.join("models", "bc_model.pth"))