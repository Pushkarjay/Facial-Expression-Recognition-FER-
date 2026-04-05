import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from utils import load_data
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CNN + SNN Hybrid Model ---
class CNN_SNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(256, num_classes)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []

        x = self.features(x)
        x = x.view(x.size(0), -1)

        for step in range(25):  # time steps
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0)


# --- Training Function ---
def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        spk_rec = model(data)
        loss = criterion(spk_rec.mean(0), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_correct += (spk_rec.mean(0).argmax(1) == targets).sum().item()
        total_samples += targets.size(0)
    return total_loss / total_samples, 100 * total_correct / total_samples


# --- Validation Function ---
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec = model(data)
            loss = criterion(spk_rec.mean(0), targets)
            total_loss += loss.item() * data.size(0)
            total_correct += (spk_rec.mean(0).argmax(1) == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / total_samples, 100 * total_correct / total_samples


# --- Main Training Loop (3 Ensemble Models) ---
if __name__ == "__main__":
    train_loader, test_loader, num_classes = load_data("data/raw", batch_size=64)
    os.makedirs("models", exist_ok=True)

    for i in range(1, 4):
        print(f"\n🔹 Training Ensemble Model {i}")
        model = CNN_SNN(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, epoch)
            test_loss, test_acc = test_model(model, test_loader, criterion)
            print(f"Epoch {epoch+1}/10 | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        torch.save(model.state_dict(), f"models/snn_model_{i}.pth")
        print(f"✅ Saved: models/snn_model_{i}.pth")
