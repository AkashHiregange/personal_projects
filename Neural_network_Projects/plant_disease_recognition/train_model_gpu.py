import os
import pandas as pd
from torchvision.io import decode_image
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F

class ImageDatasetFromFolders(Dataset):
    def __init__(self, path, transform=ToTensor):
        self.path = path
        self.class_names = os.listdir(self.path)
        self.transform = transform
        complete_dataset = []
        for dir in os.listdir(self.path):
            for image in os.listdir(os.path.join(self.path, dir)):
                complete_dataset.append((dir, os.path.join(self.path, dir, image)))
        self.complete_dataset = complete_dataset
        
    def __len__(self):
        return len(self.complete_dataset)
    
    def get_example(self, idx):
        img_path = self.complete_dataset[idx][1]
        image = Image.open(img_path)
        label = self.complete_dataset[idx][0]
        label = self.class_names.index(label)
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def scale_image(self, img):
        max_pixel = np.max(np.array(img))
        img = img/max_pixel
        return img
    
    def __getitem__(self, idx):
        img_path = self.complete_dataset[idx][1]
        image = Image.open(img_path)
        label = self.complete_dataset[idx][0]
#         print(label)
        label = self.class_names.index(label)
#         print(label)
        if self.transform:
            image = self.transform(image)
            image = self.scale_image(image)
        return image, label
    
base = os.getcwd()
train_dir = os.path.join(base,'Train')
test_dir = os.path.join(base,'Test')
val_dir = os.path.join(base,'Validation')

image_size = (256,256)
transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize(image_size)])
# transform = transforms.Compose([transforms.Resize(image_size)])
train_data = ImageDatasetFromFolders(train_dir, transform=transform)
test_data = ImageDatasetFromFolders(test_dir, transform=transform)
val_data = ImageDatasetFromFolders(val_dir, transform=transform)
print((train_data[2][0]).dtype) #check the type of the data to make sure it is torch.float32
print(train_data[1000]) #check if the data has been scaled
print(test_data.__len__())
print(val_data.__len__())

batch_size = 32
train_dl = DataLoader(train_data, batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size, shuffle=True)
val_dl = DataLoader(val_data, batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=None)
        # 2nd convolutional layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)   # your input size
            out = self._forward_features(dummy)
            flat_dim = out.numel()

        # Fully connected layer
        self.fc1 = nn.Linear(flat_dim, num_classes)
#         self.fc2 = nn.Linear(512,128)
#         self.fc3 = nn.Linear(128,num_classes)
        

    def _forward_features(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
           x: Input tensor.

        Returns:
           torch.Tensor
               The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.dropout(x, p=0.4, training=self.training) 
        x = F.relu(self.conv3(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv4(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)
        x = F.dropout(x, p=0.4, training=self.training)  
        return x
    
    def forward(self,x):
        x = self._forward_features(x)
#         x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = torch.flatten(x,1)
#         print(x.shape)
        x = self.fc1(x)            # Apply fully connected layer
#         x = F.dropout(x, p=0.4, training=self.training) 
#         x = self.fc2(x)
#         x = self.fc3(x)
        return x

model = CNN(in_channels=3, num_classes=3)
print(model)

from torchinfo import summary
print(summary(model, (32,3,256,256)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()