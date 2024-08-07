import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

class CardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(CardClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifer = nn.Linear(enet_out_size, num_classes)
    def forward(self,x):
        x = self.features(x)
        output = self.classifer(x)
        return output

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128)),
])



data_dir= '/card_dataset/train'
model = CardClassifer(num_classes=53)
dataset = CardDataset(data_dir, transform)
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_folder = '/Users/saipathivada/PycharmProjects/machinelearningbook/card_dataset/train'
valid_folder = '/Users/saipathivada/PycharmProjects/machinelearningbook/card_dataset/valid'
test_folder = '/Users/saipathivada/PycharmProjects/machinelearningbook/card_dataset/test'

train_dataset = CardDataset(train_folder, transform=transform)
val_dataset = CardDataset(valid_folder, transform=transform)
test_dataset = CardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 5
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CardClassifer(num_classes=53)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation loop'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

torch.save(model.state_dict(), 'trained_model.pth')
















