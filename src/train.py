
from random import shuffle
from torchvision.models import resnet34
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import torch
from torch import nn
from torch import optim
import numpy as np

import sys
sys.path.insert(1,'/src/utils')
from training_models import Trainer

from mnist import IndexedMnist

import matplotlib.pyplot as plt



import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = resnet34(num_classes=10,weights=None)

model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)

train_ds = MNIST("/data/mnist/",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

mnist = IndexedMnist(train_ds)

dl = DataLoader(mnist,batch_size=64,shuffle=False,num_workers=4,pin_memory=True)

N_EPOCHS = 100

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)




model = model.to(device)

trainer = Trainer(model,mnist,dl,criterion,optimizer,device)

acc_history,loss_history, fslt = trainer.first_split_train(N_EPOCHS)
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.plot(range(1,len(acc_history)+1),acc_history)
plt.title("Acurácia")
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1,len(loss_history)+1),loss_history)
plt.title("Loss")
plt.legend()

plt.savefig('./plot.png')
