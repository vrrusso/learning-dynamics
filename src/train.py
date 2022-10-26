
from torchvision.models import resnet34
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

import torch
from torch import nn
from torch import optim

import sys
sys.path.insert(1,'/src/utils')
print(sys.path)

from training_models import Trainer

import matplotlib.pyplot as plt



import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = resnet34(num_classes=10,weights=None)

model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)

train_ds = MNIST("/data/mnist/",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

train_dl = DataLoader(train_ds,batch_size=64,shuffle=True)

N_EPOCHS = 100

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)




model = model.to(device)

trainer = Trainer(model,train_ds,train_dl,criterion,optimizer,device)

acc_history,loss_history = trainer.first_split_train(N_EPOCHS)
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.plot(range(1,N_EPOCHS+1),acc_history)
plt.title("Acurácia")
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1,N_EPOCHS+1),loss_history)
plt.title("Loss")
plt.legend()

plt.savefig('./plot.png')

'''



for epoch in range(N_EPOCHS):

    start = time.perf_counter()
    # setando o modelo para treino
    model.train()


    running_corrects = 0
    running_loss = 0.0
    total_steps = len(train_dl)

    


    for i,(X,y) in enumerate(train_dl):

        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            outputs = model(X)
            _,preds = torch.max(outputs,1)

            # o input para a CrossEntropyLoss é o vetor de probabilidade dado pela rede
            # e os labels corretos em formato numérico
            loss = criterion(outputs,y.long())
            
            loss.backward()
            optimizer.step()
        
        # pondera a loss pelo tamanho do batch
        running_loss += loss.item()*y.size(0) 
        running_corrects += torch.sum(preds == y)

        if (i+1) % 200 == 0:
            print(f"Epoch: {epoch+1}/{N_EPOCHS} Steps: {i+1}/{total_steps}, Loss: {loss.item()}")




        
    epoch_acc = running_corrects.double()/len(train_ds)
    epoch_loss = running_loss/len(train_ds)

    print(f"Epoch {epoch+1} Acc:{epoch_acc} Loss:{epoch_loss}")
    print('------')
    print()
    print()
    
    stop = time.perf_counter()

    print(f"Tempo de Treinamento: {stop-start}")

        
'''
