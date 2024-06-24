import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from model import ResGRU
from train_test_valid import train_fn,valid_fn
import matplotlib.pyplot as plt
import time
import gc
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Configuration for the training script.')

    # Add arguments
    parser.add_argument('--train_seq_len', type=int, default=5, help='Training sequence length.')
    parser.add_argument('--predict_seq_len', type=int, default=1, help='Prediction sequence length.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for optimizer.')

    # Parse arguments
    args = parser.parse_args()
    
    train_valid_input = torch.load(f'dataset/torch-data/train_valid_input.pt').float()
    train_valid_label = torch.load(f'dataset/torch-data/train_valid_label.pt').float()
    train_valid_stock_number = torch.load(f'dataset/torch-data/train_valid_stock_number.pt').int()

    for i in range(train_valid_stock_number.unique().shape[0]):   
        target_stock = train_valid_stock_number.unique()[i].numpy()
        
        print(f'training {target_stock} data')
         
        # training target stock indices
        indices = torch.where(train_valid_stock_number == train_valid_stock_number.unique()[i])[0]
        train_valid_data = train_valid_input[indices]
        train_valid_target = train_valid_label[indices]

        
        # input feature scaling
        scaler = MinMaxScaler()
        train_valid_data_np = train_valid_data.numpy().reshape(-1, train_valid_data.shape[-1])
        # 初始化 scaler 並擬合訓練數據
        scaler = MinMaxScaler()
        scaler.fit(train_valid_data_np)
        dump(scaler, f'scaler/scaler{str(target_stock)}.save')
        # 對訓練和驗證數據進行轉換
        train_valid_scaled_np = scaler.transform(train_valid_data_np)
        # 將縮放後的數據重新轉換為原來的三維形狀
        train_valid_data = torch.tensor(train_valid_scaled_np.reshape(train_valid_data.shape), dtype=torch.float32)

        # label feature scaling
        column_index = 1
        data_min = scaler.data_min_[column_index]
        data_max = scaler.data_max_[column_index]
        train_valid_target_np = train_valid_target.numpy()
        train_valid_target_scaled = (train_valid_target_np-data_min)/(data_max - data_min) 
        train_valid_target = torch.tensor(train_valid_target_scaled, dtype=torch.float32)
        
        train_data = train_valid_data[:int(len(train_valid_data)*0.8)]
        valid_data = train_valid_data[int(len(train_valid_data)*0.8):]
        train_target = train_valid_target[:int(len(train_valid_target)*0.8)]
        valid_target = train_valid_target[int(len(train_valid_target)*0.8):]
        
        train_loader = DataLoader(TensorDataset(train_data, train_target), batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(valid_data, valid_target), batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = ResGRU(input_size = 16, hidden_size=[128, 64, 32], out_dim = 1, step = 5)
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay= 0.005)
        criterion = nn.MSELoss()
        best_loss_mse = float('inf')
        train_loss_curve = []
        valid_loss_curve = []

        for epoch in range(args.epochs):
            start_time = time.time()
            train_loss = train_fn(train_loader, model, criterion, optimizer, epoch, device=args.device)
            valid_loss = valid_fn(val_loader, model, criterion, args.device)  
            elapsed = time.time() - start_time

            print(
                f" Epoch {epoch+1} - avg time: {elapsed:.0f}s \n"
                f" avg_train_MSEloss: {train_loss.avg_mse:.4f} - avg_train_MAPEloss: {train_loss.avg_mape:.4f}\n"
                f" avg_val_MSEloss: {valid_loss.avg_mse:.4f} - avg_val_MAPEloss: {valid_loss.avg_mape:.4f}\n" 
            )

            train_loss_mse = train_loss.avg_mse
            valid_loss_mse = valid_loss.avg_mse 
            train_loss_curve.append(train_loss_mse)
            valid_loss_curve.append(valid_loss_mse)

            # 如果需要保存模型
            if valid_loss_mse < best_loss_mse:
                torch.save(model.state_dict(),f'model_state/state_predict_sequence{str(target_stock)}.pth')
                best_loss_mse = valid_loss_mse

        # visualize loss curve
        '''
        loss_curve_len = len(train_loss_curve)
        plt.plot(range(1,(loss_curve_len)+1), train_loss_curve, color='blue', label="train_mse")
        plt.plot(range(1,(loss_curve_len)+1), valid_loss_curve, color='red', label="valid_mse")
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Comparison of train loss and valid loss curve")
        plt.show()
        '''
        
        torch.cuda.empty_cache()
        gc.collect()