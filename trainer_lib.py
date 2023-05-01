import itertools
import time
import torch
from torch import (nn, optim)
import pandas as pd
import numpy as np

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optim.Adam(model.parameters())
        self.loss = nn.CrossEntropyLoss()
        
    def train_one_epoch(self, max_batches=None):
        device = torch.device('cuda')
        l_list = []
        acc_list = []
        for (xs, targets) in self.train_dataloader:
            self.optimizer.zero_grad()
            xs.to(device)
            targets.to(device)
            ret = self.model(xs)
            loss = self.loss(ret, targets)
            loss.backward()
            self.optimizer.step()
            l_list.append(loss.item())
            pred = ret.argmax(dim=1)
            acc = (pred==targets).float()
            acc_list.append(acc)
            
        return np.mean(l_list), np.mean(acc_list[0].mean().item())
    
    def val_one_epoch(self):
        l_list = []
        acc_list = []
        with torch.no_grad():
            for (xs, targets) in self.val_dataloader:
                ret = self.model(xs)
                loss = self.loss(ret, targets)
                l_list.append(loss.item())
                pred = ret.argmax(dim=1)
                acc = (pred==targets).float()
                acc_list.append(acc)
        return np.mean(l_list), np.mean(acc_list[0].mean().item())
                
    def train(self, epochs, max_batches=None):
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch_duration': [],
        }

        start0 = time.time()

        for epoch in range(epochs):
            start = time.time()
            
            (train_loss,train_acc) = self.train_one_epoch(max_batches = max_batches)
            (val_loss,val_acc) = self.val_one_epoch()
            # complete the following
            duration = start - time.time()
            ...

            duration = time.time() - start
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['epoch_duration'].append(duration)
            
            print("[%d (%.2fs)]: train_loss=%.2f train_acc=%.2f, val_loss=%.2f val_acc=%.2f" % (
                epoch, duration, train_loss, train_acc, val_loss, val_acc))
            
        duration0 = time.time() - start0
        print("== Total training time %.2f seconds ==" % duration0)

        return pd.DataFrame(history)
    
    def reset(self):
        self.model.apply(self._reset_parameters)
        
    def _reset_parameters(self, layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()