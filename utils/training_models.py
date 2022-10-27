from symbol import import_stmt


import time
import torch

class Trainer:

    def __init__(self,model, dataset, dataloader,criterion,optmizer,device):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optmizer
        self.device = device

    def first_split_train(self,n_epochs):
        
        acc_history = []
        loss_history = []



        start = time.perf_counter()
        for epoch in range(n_epochs):
            self.model.train()
            running_corrects = 0
            running_loss = 0.0
            total_steps = len(self.dataloader)

            for i,(X,y,idx) in enumerate(self.dataloader):

                
                X = X.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):

                    outputs = self.model(X)
                    _,preds = torch.max(outputs,1)

                    # o input para a CrossEntropyLoss é o vetor de probabilidade dado pela rede
                    # e os labels corretos em formato numérico
                    loss = self.criterion(outputs,y.long())
                
                    loss.backward()
                    self.optimizer.step()
                    
            
                # pondera a loss pelo tamanho do batch
                running_loss += loss.item()*y.size(0) 
                running_corrects += torch.sum(preds == y)

                if (i+1) % 200 == 0:
                    print(f"Epoch: {epoch+1}/{n_epochs} Steps: {i+1}/{total_steps}, Loss: {loss.item()}")

                

            epoch_acc = running_corrects.double()/len(self.dataset)
            epoch_loss = running_loss/len(self.dataset)

            

            acc_history.append(epoch_acc.cpu())
            loss_history.append(epoch_loss)

            print(f"Epoch {epoch+1} Acc:{epoch_acc} Loss:{epoch_loss}")
            print('------')
            print()
            print()
    
        stop = time.perf_counter()
        print(f"Tempo total de Treinamento: {stop-start}")

        return acc_history, loss_history