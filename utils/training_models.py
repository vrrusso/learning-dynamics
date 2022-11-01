from symbol import import_stmt



import time
import torch
import numpy as np

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


        learning_matrix = torch.zeros((n_epochs,len(self.dataset)),dtype=torch.bool)
        
        
        start = time.perf_counter()
        for epoch in range(n_epochs):
            self.model.train()
            running_corrects = 0
            running_loss = 0.0
            total_steps = len(self.dataloader)

            # para reduzir as transferências entre CPU e GPU faz uma cópia da época atual na GPU
            # e a traz para CPU apenas no fim do treino
            epoch_learning_vector = torch.zeros(len(self.dataset),dtype=torch.bool)
            epoch_learning_vector = epoch_learning_vector.to(self.device)
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
                    
                
                epoch_learning_vector[idx] = (preds == y)
                
                


                # pondera a loss pelo tamanho do batch
                running_loss += loss.item()*y.size(0) 
                running_corrects += torch.sum(preds == y)
                

                

                if (i+1) % 200 == 0:
                    print(f"Epoch: {epoch+1}/{n_epochs} Steps: {i+1}/{total_steps}, Loss: {loss.item()}")

            learning_matrix[epoch] = epoch_learning_vector.to('cpu')

            epoch_acc = running_corrects.double()/len(self.dataset)
            epoch_loss = running_loss/len(self.dataset)

            

            

            acc_history.append(epoch_acc.cpu())
            loss_history.append(epoch_loss)

            print(f"Epoch {epoch+1} Acc:{epoch_acc} Loss:{epoch_loss}")
            print('------')
            print()
            print()

            if epoch > 6 and np.mean( acc_history[-5:-1]+[acc_history[-1]]) > 0.9999:
                print("Early Stopping - Acurácia 1.0 atingida")
                break
    
        stop = time.perf_counter()
        print(f"Tempo total de Treinamento: {stop-start}")

        
        learning_matrix = learning_matrix.numpy()
        learning_matrix = learning_matrix[:epoch,:]
        reversed_learning_matrix = np.flip(learning_matrix,axis=0)

        
        fslt = np.zeros(len(reversed_learning_matrix[1]))

        for i in range(reversed_learning_matrix.shape[1]):
            false_ocurrences = np.argwhere(reversed_learning_matrix[:,i] == False)
            fslt[i] = (epoch -  false_ocurrences[0]) if len(false_ocurrences) > 0 else 0

            
        


        

        return acc_history, loss_history,fslt