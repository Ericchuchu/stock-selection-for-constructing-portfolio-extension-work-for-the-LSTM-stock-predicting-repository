import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import ResGRU
from train_test_valid import test_fn
import matplotlib.pyplot as plt
from joblib import load
from tqdm import trange


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Configuration for the training script.')

    # Add arguments
    parser.add_argument('--train_seq_len', type=int, default=5, help='Training sequence length.')
    parser.add_argument('--predict_seq_len', type=int, default=1, help='Prediction sequence length.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training.')

    # Parse arguments
    args = parser.parse_args()


    test_input = torch.load(f'dataset/torch-data/test_input.pt').float()
    test_label = torch.load(f'dataset/torch-data/test_label.pt').float()
    test_stock_number = torch.load(f'dataset/torch-data/test_stock_number.pt').int()
    train_valid_stock_number = torch.load(f'dataset/torch-data/train_valid_stock_number.pt').int()
    
    # dictionary save loss
    loss =[]
    stock_number = []
    
    print('Testing all stock')
    
    # 有些股票在2022年以前並未出現，所以以 train_valid_data 有的資料為主
    for i in range(train_valid_stock_number.unique().shape[0]):
        
        target_stock = train_valid_stock_number.unique()[i].numpy()
        
        indices = torch.where(test_stock_number == train_valid_stock_number.unique()[i])[0]
        test_data = test_input[indices]
        test_target = test_label[indices]
        
        # testing input feature scaling
        # 某些股票在 2022 年以前並無資料 2756 不包含在 train data 裡
        scaler_loaded = load(f'scaler/scaler{str(target_stock)}.save')
        test_data_np = test_data.numpy().reshape(-1, test_data.shape[-1])
        test_scaled_np = scaler_loaded.transform(test_data_np)
        test_data = torch.tensor(test_scaled_np.reshape(test_data.shape), dtype=torch.float32)
        
        # testing label feature scaling
        column_index = 1
        data_min = scaler_loaded.data_min_[column_index]
        data_max = scaler_loaded.data_max_[column_index]
        test_target_np = test_target.numpy()
        test_target_scaled = (test_target_np-data_min)/(data_max - data_min) 
        test_target = torch.tensor(test_target_scaled, dtype=torch.float32)

        test_loader = DataLoader(TensorDataset(test_data, test_target), batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        # testing model
        criterion = nn.MSELoss()
        testing_model = ResGRU(input_size = 16, hidden_size=[128, 64, 32], out_dim = 1, step = 5)
        testing_model.load_state_dict(torch.load(f'model_state/state_predict_sequence{str(target_stock)}.pth'))
        test_loss, true, pred = test_fn(test_loader, testing_model, criterion, args.device) 
        stock_number.append(str(target_stock ))
        true = np.concatenate([arr.ravel() for arr in true])
        pred = np.concatenate([arr.ravel() for arr in pred])
        loss.append(test_loss.avg_mse)
        pred_df = pd.DataFrame({'predicted_price':pred,'true_price':true})
        file_name = f'predicted_result/{str(target_stock)} 2023 predicted prices.csv'
        pred_df.to_csv(file_name, mode='w', header=True, index=False)

        '''
        plot_len = len(true)
        plt.plot(np.arange(plot_len), true, color='blue', label="True Price")
        plt.plot(np.arange(plot_len), pred, color='red', label="Estimated Price")
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.title("Comparison of Estimated Price and True Price")
        plt.show()
        '''
    

        torch.cuda.empty_cache()
        gc.collect()
        
    loss_df = pd.DataFrame({'stock_number':stock_number,'test_mse_loss':loss})
    loss_df.to_csv(f'test_loss/test_loss.csv')