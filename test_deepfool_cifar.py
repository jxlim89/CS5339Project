import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



net = Net()
net.load_state_dict(torch.load("./models/jb_cifar.pt")) #model with Jacobian

net2 = Net()
net2.load_state_dict(torch.load("./models/basic_cifar.pt")) #load the model without Jacobian regul

# Switch to evaluation mode
net.eval()
net2.eval()

im_orig = Image.open('./images/test1.jpg')


mean = [ 0.1307,0.1307,0.1307 ] #Mean and std have to be transformed using the same values for training models
std = [ 0.3081,0.3081,0.3081 ]

# Remove the mean by normalizing
im = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

print(im.shape)

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)
labels = open(os.path.join('./models/cifar10_label.txt'), 'r').read().split('\n')

print (label_orig,label_pert, loop_i )
print("Perturbed:" , np.linalg.norm(r))
str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage()])

r2, loop_i2, label_orig2, label_pert2, pert_image2 = deepfool(im, net2)
print("Perturbed2:" , np.linalg.norm(r2))

str_label_pert2 = labels[np.int(label_pert2)].split(',')[0]
str_label_orig2 = labels[np.int(label_orig2)].split(',')[0]

print("Original label2 = ", str_label_orig2)
print("Perturbed label2 = ", str_label_pert2)


plot2 = plt.figure(1)
plt.imshow(tf(im).convert('L'))
plt.title("Original\n " + str_label_orig)

plot1 = plt.figure(2)
plt.imshow(tf(pert_image.cpu()[0]).convert('L'))
plt.title("Jacobian Regularization \n"+ "Perturbed:" + str(np.linalg.norm(r)) + "\n"+str_label_pert)

plot3 = plt.figure(3)
plt.imshow(tf(pert_image2.cpu()[0]).convert('L'))
plt.title("No Jacobian Regularization \n"+ "Perturbed:" + str(np.linalg.norm(r2)) + "\n"+str_label_pert2)

plt.show()