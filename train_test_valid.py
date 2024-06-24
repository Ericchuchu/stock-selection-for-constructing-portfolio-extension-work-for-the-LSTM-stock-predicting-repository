
import time
import warnings
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.mse = 0
        self.mape = 0
        self.avg_mse = 0 
        self.avg_mape = 0
    
    def update(self, val, mse, mape, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mse += mse
        self.mape += mape
        self.avg_mse = self.mse / self.count
        self.avg_mape = self.mape / self.count
        self.avg = self.sum / self.count


# 計算MSE
def MSE(true, predicted):
    return np.mean((true - predicted) ** 2)

# 計算MAPE
def MAPE(true, predicted):
    true_o = true
    pred_o = predicted
    mape = np.mean(np.abs((true_o - pred_o) / (true_o + 1e-10))) * 100
    return mape


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler=None, device='cuda'):
    model.train()
    losses = AverageMeter()
    start = time.time()
    global_step = 0
    total_steps = len(train_loader)
    
    with tqdm(total=total_steps, desc=f"Epoch {epoch+1}", leave=True, ncols=100, unit='step') as pbar:
        for step, (inputs, labels) in enumerate(train_loader):
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            labels = labels.detach().to('cpu').numpy()            
            y_pred = y_pred.detach().to('cpu').numpy()
            MSE_loss = MSE(labels, y_pred)
            MAPE_loss = MAPE(labels, y_pred)
            losses.update(loss.item(), MSE_loss, MAPE_loss)
            
            '''print(
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}]"
                f" Elapsed: {(time.time()-start):.0f}s"
                f" Loss: {losses.val:.4f}"          
            )'''

            pbar.update(1)  

    return losses


def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    start = time.time()
    preds = []
    true = []
    
    for step, (inputs, labels) in enumerate(valid_loader):
        batch_size = 64
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            y_pred = model(inputs)

        loss = criterion(y_pred, labels)  
        labels = labels.detach().to('cpu').numpy()            
        y_pred = y_pred.detach().to('cpu').numpy()
        true.append(labels)
        preds.append(y_pred)
        MSE_loss = MSE(labels, y_pred)
        MAPE_loss = MAPE(labels, y_pred)
        losses.update(loss.item(), MSE_loss, MAPE_loss)

    return losses

def test_fn(test_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    start = time.time()
    preds = []
    true = []
    print("Testing the model")

    for step, (inputs, labels) in enumerate(test_loader):
        batch_size = 64
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            y_pred = model(inputs)

        loss = criterion(y_pred, labels)  
        labels = labels.detach().to('cpu').numpy()            
        y_pred = y_pred.detach().to('cpu').numpy()
        true.append(labels)
        preds.append(y_pred)
        MSE_loss = MSE(labels, y_pred)
        MAPE_loss = MAPE(labels, y_pred)
        losses.update(loss.item(), MSE_loss, MAPE_loss)

    return losses,true,preds
        