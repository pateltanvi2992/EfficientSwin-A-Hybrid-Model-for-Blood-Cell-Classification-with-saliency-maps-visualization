import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import medmnist
from medmnist import BloodMNIST, INFO
from torchvision import transforms
import timm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CustomHybridModel class (EfficientSwin trained model)
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

# Load the BloodMNIST test dataset
test_dataset = BloodMNIST(split='val', download=True, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the saved model
# please adjust the path of the model
saved_model_path = "/content/bloodmnist_EfficientSwin_best_model.pt"
model = CustomHybridModel(num_classes=n_classes)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Function to extract features from both Swin and EfficientNet components of the model
def extract_features_and_labels(model, dataloader, device='cpu'):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Extract features from the Swin and EfficientNet components
            swin_features = model.swin(inputs)
            efficientnet_features = model.efficientnet(inputs)
            combined_features = torch.cat((swin_features, efficientnet_features), dim=1)
            all_features.append(combined_features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_features), np.concatenate(all_labels)  # Combine feature batches and labels into arrays

# Extract features and labels using the trained model
features, labels = extract_features_and_labels(model, test_loader, device=device)
# Reshape labels to 1D
labels = labels.flatten()  # or labels = labels.squeeze()
print(labels)

# Calculate centroids for each class using the true labels
class_centroids = np.zeros((n_classes, features.shape[1]))
for i in range(n_classes):
    class_features = features[labels == i]
    class_centroids[i] = class_features.mean(axis=0)

# Calculate distances between class centroids
centroid_distances = cdist(class_centroids, class_centroids, metric='euclidean')
min_distance = np.min(centroid_distances[np.nonzero(centroid_distances)])
max_distance = np.max(centroid_distances)

print(f"Minimum Distance between Class Centroids: {min_distance}")
print(f"Maximum Distance between Class Centroids: {max_distance}")

# Reduce dimensions to 2D using PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)
reduced_centroids = pca.transform(class_centroids)

# Plot the clusters with class names
plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.6)

# Plot centroids with class labels
for i, label in enumerate(classes.values()):
    plt.scatter(reduced_centroids[i, 0], reduced_centroids[i, 1], marker='x', s=200, color='red')
    #plt.text(reduced_centroids[i, 0], reduced_centroids[i, 1], label, fontsize=12, fontweight='bold')

plt.title('Class Centroid Visualization with PCA and Class Names')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label=labels.all)
plt.show()