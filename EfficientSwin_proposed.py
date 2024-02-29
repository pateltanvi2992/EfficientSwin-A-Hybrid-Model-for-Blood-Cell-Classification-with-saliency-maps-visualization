import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from tqdm import tqdm
import timm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a model on MedMNIST')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=35, help='number of epochs to train (default: 35)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--save_fig', type=str, default='training_validation_accuracy.png', help='filename to save the figure (default: "training_validation_accuracy_b0.png")')
args = parser.parse_args()

# Set random seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# Dataset information
data_flag = 'bloodmnist'
info = INFO[data_flag]
n_classes = len(info['label'])
classes = info['label']
BloodMNIST = getattr(medmnist, info['python_class'])

# Preprocessing
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load datasets
train_dataset = BloodMNIST(split="train", download=True, transform=data_transform)
val_dataset = BloodMNIST(split="val", download=True, transform=data_transform)
test_dataset = BloodMNIST(split='test', transform=data_transform, download=True)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Custom Hybrid Model
class CustomHybridModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomHybridModel, self).__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        swin_feature_size = self.swin.num_features
        efficientnet_feature_size = self.efficientnet.num_features

        self.classifier = nn.Linear(swin_feature_size + efficientnet_feature_size, num_classes)

    def forward(self, x):
        swin_features = self.swin(x)
        efficientnet_features = self.efficientnet(x)
        combined = torch.cat((swin_features, efficientnet_features), dim=1)
        return self.classifier(combined)

# Initialize model
model = CustomHybridModel(num_classes=n_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Directories for saving outputs
os.makedirs("models", exist_ok=True)

# Evaluation Function
def evaluate_model(loader):
    model.eval()
    total_correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * total_correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=classes))
    return accuracy

# Training loop
best_val_accuracy = 0
early_stopping_patience = 5
no_improvement_epochs = 0
train_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    epoch_loss = train_loss / len(train_loader)
    epoch_accuracy = 100 * train_correct / train_total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch {epoch + 1}/{args.epochs}: Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')

    val_accuracy = evaluate_model(val_loader)
    val_accuracies.append(val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improvement_epochs = 0
        torch.save(model.state_dict(), f"models/{data_flag}_best_model.pth")
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

# Plotting
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.savefig(args.save_fig)
plt.show()

# Final evaluation on test set
print('Evaluating on Test Set...')
test_accuracy = evaluate_model(test_loader)
