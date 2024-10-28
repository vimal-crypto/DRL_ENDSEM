import os
import random
from PIL import Image
import torch
import torch.nn as nn
from Convert import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Custom Dataset to load images from LFW
class LFWCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, test_fraction=0.2):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}

        for idx, label in enumerate(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                self.label_to_idx[label] = idx
                images = [os.path.join(label_path, img_name) for img_name in os.listdir(label_path)]

                # Select 20% of the images for testing
                num_test_images = int(len(images) * test_fraction)
                self.image_paths += random.sample(images, num_test_images)
                self.labels += [idx] * num_test_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        while True:
            try:
                image = Image.open(img_path).convert('RGB')
                break  # Break the loop if image is loaded successfully
            except Exception as e:
                idx = (idx + 1) % len(self.image_paths)  # Move to the next image
                img_path = self.image_paths[idx]
                if idx == 0:  # Prevent infinite loop
                    raise ValueError("All images are corrupted or unreadable.")

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Load the ResNet18 model
resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5749)
f1_rc = f1()
# Load the state_dict (weights) from the saved model
resnet_model.load_state_dict(torch.load("resnet18_finetuned.pth"))
recal = recall()
# Modify the last fully connected layer to match the number of classes in your current dataset (e.g., 40)
num_classes = 40
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

# Set the model to evaluation mode
resnet_model.eval()
preci = precition()
# Dataset directory and transformation
root_dir = 'C:/Users/vimal/PycharmProjects/Reinforcement_Learning/End_Sem_DRL/archive/lfw-deepfunneled/lfw-deepfunneled'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the test dataset and DataLoader with 20% of the data
test_dataset = LFWCustomDataset(root_dir=root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the ResNet18 model with augmentations during inference
true_labels = []
pred_labels = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet_model.to(device)
acc= accuracy()
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # Get original predictions
    outputs = resnet_model(images)
    _, preds = torch.max(outputs, 1)

    true_labels.extend(labels.cpu().numpy())
    pred_labels.extend(preds.cpu().numpy())

# # Calculate metrics
accr = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1_src = f1_score(true_labels, pred_labels, average='weighted')
print(f"ResNet18 Model Evaluation (using 20% test set): \nAccuracy: {acc:.4f}, Precision: {preci:.4f}, Recall: {recal:.4f}, F1-score: {f1_rc:.4f}")
