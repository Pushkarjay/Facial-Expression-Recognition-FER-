import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------
# CNN + SNN Hybrid Model Definition
# ------------------------------------------------
class CNN_SNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN_SNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 48x48 → 48x48
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 48x48 → 24x24
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)               # 24x24 → 12x12
        )
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ------------------------------------------------
# Ensemble Prediction Function (FIXED)
# ------------------------------------------------
def ensemble_predict(models, images, device):
    """Run ensemble prediction across multiple models"""
    with torch.no_grad():
        preds = []
        for model in models:
            outputs = model(images.to(device))
            preds.append(F.softmax(outputs, dim=1))

        avg_output = torch.mean(torch.stack(preds), dim=0)
        return avg_output  # ✅ return full tensor (no .item())
