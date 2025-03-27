from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import DemonstrationDataset
from src.policy_network import PolicyNetwork
import os, time, torch, torch.nn as nn, torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_bc():
    train_set = DemonstrationDataset("data/train")
    val_set = DemonstrationDataset("data/val")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, pin_memory=True)

    model = PolicyNetwork(num_actions=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 5
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_entropy_sum = 0.0
        num_samples = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Training]", leave=False)
        for states, actions in train_bar:
            states, actions = states.to(device), actions.to(device)
            logits = model(states)
            loss = criterion(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_current = states.size(0)
            train_loss_sum += loss.item() * batch_size_current
            num_samples += batch_size_current

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            train_entropy_sum += entropy.item() * batch_size_current

            train_bar.set_postfix({
                "Batch Loss": f"{loss.item():.3f}",
                "Avg Loss": f"{train_loss_sum/num_samples:.3f}",
                "Entropy": f"{entropy.item():.3f}"
            })

        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        avg_train_entropy = train_entropy_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Validation]", leave=False)
        for states, actions in val_bar:
            states, actions = states.to(device), actions.to(device)
            logits = model(states)
            loss = criterion(logits, actions)
            val_loss_sum += loss.item() * states.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == actions).sum().item()
            total += states.size(0)
            val_bar.set_postfix({"Batch Loss": f"{loss.item():.3f}"})

        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc = correct / total
        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch+1}/{n_epochs} Summary:\n"
            f"  Time: {epoch_time:.1f}s\n"
            f"  Train Loss: {avg_train_loss:.3f}, Entropy: {avg_train_entropy:.3f}\n"
            f"  Val Loss: {avg_val_loss:.3f}, Accuracy: {val_acc:.3f}\n"
        )

    os.makedirs("models", exist_ok=True)
    sample_state = torch.rand(1, 1, 84, 84, device=device)
    torch.onnx.export(model, sample_state, "models/bc_model.onnx", export_params=True, opset_version=17, do_constant_folding=True)
    torch.save(model.state_dict(), os.path.join("models", "bc_model.pth"))

if __name__ == "__main__":
    train_bc()