import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import medmnist
import timm
from medmnist import BloodMNIST, INFO
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Custom Model
class CustomSwinModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomSwinModel, self).__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

        # Freeze all layers in the network
        for param in self.swin.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers. The exact layers to unfreeze depend on the model architecture.
        for param in self.swin.head.parameters():
            param.requires_grad = True

        # modify the classifier (head):
        num_features = self.swin.head.in_features
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.swin(x)

# Initialize model
model = CustomSwinModel(num_classes=n_classes)

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Directories for saving outputs
os.makedirs("models", exist_ok=True)

# Implement early stopping mechanism
early_stopping_patience = 5
no_improvement_epochs = 0
best_val_accuracy = 0

# Evaluation Function (Corrected)
def evaluate_model(loader):
    model.eval()
    total_correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(f'Confusion Matrix: \n{confusion_matrix(y_true, y_pred)}')
    print(f'Classification Report: \n{classification_report(y_true, y_pred, target_names=classes)}')
    print(total_correct)
    print(total)
    accuracy = 100 * total_correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy, y_true, y_pred

NUM_EPOCHS = 35

# Training and Validation Metrics
train_losses = []
train_accuracies = []
test_accuracies = []
# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        labels = labels.squeeze().long()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Compute training metrics
    epoch_loss = train_loss / len(train_loader)
    epoch_accuracy = 100 * train_correct / train_total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)


    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}: Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')
    # Evaluate on validation set (instead of test set)
    val_accuracy, _, _ = evaluate_model(val_loader)
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}: Validation Accuracy: {val_accuracy}')
    test_accuracies.append(val_accuracy)

    # Check for early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improvement_epochs = 0
        # Save the best model
        best_model_state_dict = model.state_dict()
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

# Save the best model
best_model_filename = f"models/{data_flag}_best_model.pt"
torch.save(best_model_state_dict, best_model_filename)

# Plotting with dots at each epoch
plt.figure(figsize=(12, 6))

# Plot for Training Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linestyle='-', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.xticks(range(1, len(train_losses) + 1))  # Ensure x-axis ticks match the number of epochs
plt.legend()

# Plot for Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', linestyle='-', marker='o')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Validation Accuracy', linestyle='-', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.xticks(range(1, max(len(train_accuracies), len(test_accuracies)) + 1))  # Ensure x-axis ticks match the number of epochs
plt.legend()
plt.savefig("swinTransformer.png")
plt.show()

# Evaluate on training and test sets
print('==> Evaluating on Training Set...')
train_accuracy, y_true, y_pred = evaluate_model(train_loader)
print(f'Training Set Accuracy: {train_accuracy:.2f}%')
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

print('==> Evaluating on Test Set...')
test_accuracy, y_true, y_pred = evaluate_model(val_loader)
print(f'Test Set Accuracy: {test_accuracy:.2f}%')
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))
