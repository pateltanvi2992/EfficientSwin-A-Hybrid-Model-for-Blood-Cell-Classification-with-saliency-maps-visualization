import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the best trainied model
model.load_state_dict(torch.load(best_model_filename))
model.to(device)
model.eval()

# Select one image and its label dataset
# Here test_loader is the test set
images, labels = next(iter(test_loader))
labels = labels.squeeze().long()

# Move the images and labels to the device
images, labels = images.to(device), labels.to(device)
print(f"Image shape: {images.shape}")
print(f"Label shape: {labels.shape}")

# Select only the first image and its label
image = images[0:1].to(device)  # Select only the first image
label = labels[0].unsqueeze(0).to(device)  # Select the first label and maintain it as a batch

# Set requires_grad to True to compute gradients with respect to the input image
image.requires_grad_()

# Forward pass
output = model(image)

# Compute the loss
loss = criterion(output, label)
model.zero_grad()
loss.backward()

# Saliency is the absolute value of the gradient
saliency = image.grad.data.abs().squeeze()

# Convert the saliency tensor to a numpy array and take the maximum across the channels
saliency_map = saliency.cpu().numpy().max(axis=0)

# Normalize the original image for display
image_display = images[0].cpu().numpy().transpose(1, 2, 0)
image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min())

# Overlay the saliency map on the original image
plt.imshow(image_display)
plt.imshow(saliency_map, cmap=plt.cm.hot, alpha=0.5)  # Alpha controls the transparency
plt.axis('off')
plt.savefig('saliency_map_overlay.png')  # Save the figure
plt.show()
