import torch
import gym
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from PIL import Image
import os

class LFWCustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}

        for idx, label in enumerate(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                self.label_to_idx[label] = idx
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset directory and transformations (same as DL model)
root_dir = 'C:/Users/vimal/PycharmProjects/Reinforcement_Learning/End_Sem_DRL/archive/lfw-deepfunneled/lfw-deepfunneled'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize back to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



lfw_dataset = LFWCustomDataset(root_dir=root_dir, transform=transform)

# Load a pre-trained ResNet18 model as feature extractor
model = models.resnet18(pretrained=True)
num_classes = len(lfw_dataset.label_to_idx)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Custom environment for face recognition using ResNet18
class FaceRecognitionEnv(gym.Env):
    def __init__(self, model, dataset, device):
        super(FaceRecognitionEnv, self).__init__()
        self.model = model
        self.dataset = dataset
        self.device = device
        self.current_idx = 0

        # Action space: 2 actions (recognize or not recognize)
        self.action_space = gym.spaces.Discrete(2)
        # Observation space: shape of an image (3, 224, 224)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32)

    def step(self, action):
        image, label = self.dataset[self.current_idx]
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            logits = self.model(image)
            pred = logits.argmax(dim=1).item()

        reward = 1 if action == pred else -1
        done = True
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        return np.array(image.cpu().squeeze()), reward, done, {}

    def reset(self):
        self.current_idx = 0
        image, _ = self.dataset[self.current_idx]
        return np.array(image)

# Instantiate the environment using ResNet18 as feature extractor
env = DummyVecEnv([lambda: FaceRecognitionEnv(model, lfw_dataset, device)])

# Define DQN model using the CNN feature extractor
dqn_model = DQN('CnnPolicy', env, verbose=1, buffer_size=5000, policy_kwargs={'normalize_images': False})

# Train the DQN model for 10,000 timesteps (roughly equivalent to 10 epochs)
dqn_model.learn(total_timesteps=50000)

# Save the trained DRL model
dqn_model.save("drl_face_recognition")
print("DRL Model saved successfully!")
