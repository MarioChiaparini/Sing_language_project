from __future__ import print_function
import cv2
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset  import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


sinais =  {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }


class SingLanguageData(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        if self.train == True:
            self.signs_lang_dataset = pd.read_csv('/Users/mariochiaparini/Desktop/kaggles_machine_learning/data/sign_mnist_train.csv')
        else:
            self.signs_lang_dataset = pd.read_csv('/Users/mariochiaparini/Desktop/kaggles_machine_learning/data/sign_mnist_test.csv')
        self.X_set = self.signs_lang_dataset.iloc[:, 1:].values
        self.y_set = self.signs_lang_dataset.iloc[:, 0].values
        
        self.X_set = np.reshape(self.X_set, (self.X_set.shape[0], 1, 28, 28)) / 255
        self.y_set = np.array(self.y_set)
    def __getitem__(self, index):
        image = self.X_set[index, :, :]
        label = self.y_set[index]
        sample = {'Sinais':image,'label':label}
        return sample
    def __len__(self):
        return self.X_set.__len__()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 100, kernel_size=5)
        self.conv2 = nn.Conv2d(100,80, kernel_size=5)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride = 2, padding = 0)
        
        self.batch_norm1 = nn.BatchNorm2d(100)
        self.batch_norm2 = nn.BatchNorm2d(80)
        
        self.fc1 = nn.Linear(1280, 250)
        self.fc2 = nn.Linear(250, 25)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = F.log_softmax(x, dim=1)
        
        return x


def train(model, optimizer, epoch, device, train_loader, log_interval):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        
        img = data['Sinais']
        img = img.type(torch.FloatTensor).to(device)
        target = data['label']
        target = target.type(torch.LongTensor).to(device)
               
        optimizer.zero_grad()
        
        output = model(img)
        #print(output.shape)
        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
def test(model, device, test_loader):
    model.eval()
    test_loss = 0 
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            
            img = data['Sinais']
            img = img.type(torch.FloatTensor).to(device)
            target = data['label']
            target = target.type(torch.LongTensor).to(device)
            
            out = model(img)
            test_loss += F.nll_loss(out, target).item() 
            pred = out.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

batch_size_train = 5
batch_size_test = 4

dataset_train = SingLanguageData(train = True)
dataset_test = SingLanguageData(train = False)
train_loader = DataLoader(dataset = dataset_train, batch_size = batch_size_train)
test_loader = DataLoader(dataset = dataset_test, batch_size = batch_size_test)


torch.manual_seed(123)

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

learning_rate = 0.001
num_epochs = 7
model = Network()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.2, weight_decay = 0.002)

log_interval = 27455

for epoch in range(1, num_epochs + 1):
    train(model, optimizer, epoch, device, train_loader, log_interval)
    test(model, device, test_loader)

torch.save(model, '/Users/mariochiaparini/Desktop/kaggles_machine_learning/computervision/libras.pt')