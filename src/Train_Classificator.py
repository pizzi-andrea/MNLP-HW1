import torch
import pandas as pd
import time
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from tqdm.auto import tqdm
from pathlib import PosixPath
from torch.utils.data import DataLoader
from torch.optim import Optimizer

class Train_Classificator:
    """
    Generic Class use to train pytorch model on `Cultural_Dataset`
    """
    def __init__(self, model, loss_fn:torch.nn.CrossEntropyLoss, optimizer:Optimizer) -> None:
        """
        Initialize Train class

        """
        self.__loss = loss_fn
        self.__model = model 
        self.__optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.__accuracy = MulticlassAccuracy(num_classes=3).to(self.device)
        self.__f1 = MulticlassF1Score(num_classes=3).to(self.device)
        self.__precision = MulticlassPrecision(num_classes=3).to(self.device)
        self.__auc = MulticlassAUROC(num_classes=3).to(self.device)
        self.__recall = MulticlassRecall(num_classes=3).to(self.device)
        
    def fit(self,
            train_loader: DataLoader,
            out_folder: PosixPath,
            epochs: int = 1) -> pd.DataFrame:
        """
        Fit the model e restituisce un DataFrame con i seguenti metrics per ogni epoch:
        - loss
        - accuracy
        - recall
        - precision
        - F1
        - AUC
        """
        # Sposto model e criteri su device
        self.__model   = self.__model.to(self.device)
        self.__loss    = self.__loss.to(self.device)
        self.__accuracy = self.__accuracy.to(self.device)
        self.__recall   = self.__recall.to(self.device)
        self.__precision = self.__precision.to(self.device)
        self.__f1       = self.__f1.to(self.device)
        self.__auc      = self.__auc.to(self.device)

        # Preparazione output folder
        out_folder.mkdir(exist_ok=True)

        # Storage dei metrics
        metrics = {
            'loss': [],
            'accuracy': [],
            'recall': [],
            'precision': [],
            'F1': [],
            'AUC': []
        }

        for epoch in range(1, epochs+1):
            # Azzeramento dei metrics
            self.__accuracy.reset()
            self.__recall.reset()
            self.__precision.reset()
            self.__f1.reset()
            self.__auc.reset()

            epoch_loss = 0.0
            n_batches = 0

            self.__model.train()
            for X, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=True):
                # 1) Preparo i batch su device e nel tipo corretto
                X = X.to(self.device).float()
                y = y.to(self.device).long()

                # 2) Azzeramento gradienti
                self.__optimizer.zero_grad()

                # 3) Forward
                logits = self.__model(X)

                # 4) Calcolo loss
                loss = self.__loss(logits, y)   # ora loss è un tensor float32

                # 5) Backward + step
                loss.backward()
                self.__optimizer.step()

                # 6) Aggiorno i metrics streaming
                self.__accuracy.update(logits, y)
                self.__recall.update(logits, y)
                self.__precision.update(logits, y)
                self.__f1.update(logits, y)
                self.__auc.update(logits, y)

                # 7) Accumulo loss scalare
                epoch_loss += loss.item()
                n_batches += 1

            # Fine epoch: calcolo valori medi
            avg_loss = epoch_loss / n_batches
            metrics['loss'].append(avg_loss)
            metrics['accuracy'].append(self.__accuracy.compute().item())
            metrics['recall'].append(self.__recall.compute().item())
            metrics['precision'].append(self.__precision.compute().item())
            metrics['F1'].append(self.__f1.compute().item())
            metrics['AUC'].append(self.__auc.compute().item())

        # Salvo i results
        df = pd.DataFrame(metrics)
        df.to_csv(out_folder / 'metrics.csv', index=False)

        # Checkpoint finale
        ckpt = {
            'epoch': epochs,
            'model_state_dict': self.__model.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }
        torch.save(ckpt, out_folder / f'checkpoint_{time.strftime("%Y%m%d_%H%M%S")}.pth')

        return df
    
    def get_model(self) -> torch.nn.Module:
        """
        Return the model
        """
        return self.__model
    
    def fit_and_get(self, train_dataset:DataLoader, out_folder:PosixPath, epochs:int=1) -> torch.nn.Module:
        """
        Fit current model and return fitted model
        """
        self.fit(train_dataset, out_folder, epochs)
        return self.get_model()