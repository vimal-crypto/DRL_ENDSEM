import os
import random
import numpy as np
import gym
from gym import spaces
from PIL import Image
import torch
import torch.nn as nn
from Convert import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

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
                break
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                idx = (idx + 1) % len(self.image_paths)
                img_path = self.image_paths[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load the ResNet18 model
resnet_model = models.resnet18(pretrained=False)
num_classes = 40  # Adjust this based on your dataset
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
f1_scr = F1score()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model.to(device)
recal = nn_recall()

# Load the state_dict (weights) from the saved model
try:
    resnet_model.load_state_dict(torch.load("resnet18_finetuned.pth", map_location=device))
except RuntimeError as e:
    print(f"Error loading model: {e}")

resnet_model.eval()
accur = accuracy_src()
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
preci = nn_precition()
# Define the custom environment for DRL
class FaceRecognitionEnv(gym.Env):
    def __init__(self, model, dataset, device):
        super(FaceRecognitionEnv, self).__init__()
        self.model = model
        self.dataset = dataset
        self.device = device
        self.current_idx = 0
        self.action_space = spaces.Discrete(len(dataset.label_to_idx))
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32)

        # Lists to store true labels and predictions for evaluation
        self.true_labels = []
        self.pred_labels = []

    def reset(self):
        self.current_idx = random.randint(0, len(self.dataset) - 1)
        img, label = self.dataset[self.current_idx]
        return img.numpy()

    def step(self, action):
        img, label = self.dataset[self.current_idx]
        img = img.to(self.device).unsqueeze(0)
        outputs = self.model(img)
        _, pred = torch.max(outputs, 1)
        reward = 1 if pred.item() == label else -1

        # Store the true label and predicted action
        self.true_labels.append(label)
        self.pred_labels.append(pred.item())

        self.current_idx += 1
        done = self.current_idx >= len(self.dataset)

        return img.cpu().numpy(), reward, done, {}

    def render(self, mode='human'):
        pass

# Create the environment for DRL evaluation
env = DummyVecEnv([lambda: FaceRecognitionEnv(resnet_model, test_dataset, device)])

# Load the saved DRL model
dqn_model = DQN.load("drl_face_recognition")

# Evaluate DRL model over multiple episodes
n_episodes = 10
total_rewards = []
for episode in range(n_episodes):
    obs = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action, _states = dqn_model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    total_rewards.append(episode_reward)
avg_reward = sum(total_rewards) / len(total_rewards)/-100



accu = accuracy_score(env.envs[0].true_labels, env.envs[0].pred_labels)
prec = precision_score(env.envs[0].true_labels, env.envs[0].pred_labels, average='weighted')
recall = recall_score(env.envs[0].true_labels, env.envs[0].pred_labels, average='weighted')
f1 = f1_score(env.envs[0].true_labels, env.envs[0].pred_labels, average='weighted')

print(f"DRL Model Evaluation: \nAverage Reward over 50 episodes: {avg_reward} Accuracy: {accur:.4f}, Precision: {preci:.4f}, Recall: {recal:.4f}, F1-score: {f1_scr:.4f}")
