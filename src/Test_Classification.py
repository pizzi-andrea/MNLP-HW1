import torch
import os
import pandas as pd
import time
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
from pathlib import PosixPath
from torch.utils.data import DataLoader
from torch.optim import Optimizer

class Eval_Classificator:
    """
    Generic Class use to Evaluate pytorch model on `Cultural_Dataset`
    """
    def __init__(self, model, loss_fn) -> None:
        """
        Initialize Train class

        """
        self.__loss = loss_fn
        self.__model = model 
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.__accuracy = MulticlassAccuracy(num_classes=3).to(self.device)
        self.__f1 = MulticlassF1Score(num_classes=3).to(self.device)
        self.__precision = MulticlassPrecision(num_classes=3).to(self.device)
        self.__auc = MulticlassAUROC(num_classes=3).to(self.device)
        self.__recall = MulticlassRecall(num_classes=3).to(self.device)
        
    def eval(self, train_dataset:DataLoader, out_folder:PosixPath, epochs:int=1) -> pd.DataFrame:
        """
        Fit the `model` and compute standard accuracy metrics:
            - `Accuracy`:
            - `Recall`:
            - `Precision`:
            - `F1 Score`:
        """
        metrics = {
            'loss' :   [],
            'accuracy':[],
            'F1':      [],
            'recall':  [],
            'precision': [],
            'AUC': []
        }

        if not out_folder.exists():
            os.mkdir(out_folder)

        with torch.no_grad():
            for epoch in  range(epochs):
                # init batch metrics
                batch_loss = 0.0
                num_batches = 0
                # Init Metrics
                self.__accuracy.reset()
                self.__f1.reset()
                self.__recall.reset()
                self.__auc.reset()
                self.__precision.reset()

                # each element (sample) in train_dataset is a batch
                for step, (X, y) in tqdm(enumerate(train_dataset), desc="Batch", leave=False):
                    # inputs in the batch
                    inputs = X.to(self.device)
                    # outputs in the batch
                    targets = y.to(self.device)

                    # When you're using negative sampling, your model should not return full output_distribution (logits over vocab)
                    # Instead, it should return the loss directly (as we did in SkipGram.forward())
                    
                    loss = self.__loss(inputs, targets) 
                    out = self.__model(inputs) 

                    self.__accuracy.update(out, y)
                    self.__recall.update(out, y)
                    self.__f1.update(out, y)
                    self.__precision.update(out,y)
                    self.__auc.update(out, y)

                    batch_loss += loss
                    num_batches += 1
                
            batch_loss /=num_batches
            metrics.get('loss', []).append(batch_loss)
            metrics.get('accuracy',[]).append(self.__accuracy.compute())
            metrics.get('recall',[]).append(self.__recall.compute())
            metrics.get('f1',[]).append(self.__f1.compute())
            metrics.get('precision',[]).append(self.__precision.compute())
            metrics.get('AUC',[]).append(self.__accuracy.compute())
        
        r = pd.DataFrame(metrics)
        r.to_csv(out_folder.joinpath('metrics.csv'))
        
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.__model.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
            'loss': self.__loss,
        }, out_folder.joinpath(f'checkpoint{time.asctime(time.gmtime())}.pth'))

        return r