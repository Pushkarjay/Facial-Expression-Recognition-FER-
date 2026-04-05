import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import CNN_SNN, ensemble_predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

test_data = datasets.ImageFolder(root="data/raw/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model_paths = [
    "models/snn_model_1.pth",
    "models/snn_model_2.pth",
    "models/snn_model_3.pth"
]

models = []
for path in model_paths:
    model = CNN_SNN(num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.eval()
    models.append(model)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = ensemble_predict(models, images, device)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n✅ Ensemble Accuracy: {100 * correct / total:.2f}% on Test Dataset\n")
