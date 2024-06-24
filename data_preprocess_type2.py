import glob
import pandas as pd
import numpy as np
import torch
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load


for part in range(28):
    print(f'dealing with preprocessed data part {part}')
    merged_df = pd.read_excel(f'preprocessed_data/preprocessed_data_part{part}.xlsx')
    # train data and valid data   
    merged_df['日期'] = pd.to_datetime(merged_df['日期'])
    merged_df.drop(columns=['股票名稱'], inplace = True)
    start_date = '2022-01-04'
    end_date = '2022-12-30'
    test_data = merged_df[(merged_df['日期'] >= start_date) & (merged_df['日期'] <= end_date)].reset_index(drop=True)
    train_data = merged_df[(merged_df['日期'] < start_date)].reset_index(drop=True)
    torch.save(test_data, f'dataset/torch-data/raw_test_data{part}.pt') # save raw data
    train_data.dropna(inplace=True)
    train_data.drop(columns=['日期'], inplace=True)
    test_data.dropna(inplace=True)
    test_data.drop(columns=['日期'], inplace=True)


    # minmax scaler
    train_scaler = MinMaxScaler(feature_range=(0,1))
    # Isolate the column to keep unchanged
    stock_number_column = train_data['股票代號'].copy()
    # Drop the column from the DataFrame
    train_data.drop(columns = ['股票代號'], inplace=True)
    train_data = pd.DataFrame(train_scaler.fit_transform(train_data), columns = train_data.columns, index = train_data.index)
    train_data['股票代號'] = stock_number_column

    # 保存scaler
    dump(train_scaler, 'scaler.save')
    test_scaler = load('scaler.save')

    # Isolate the column to keep unchanged
    stock_number_column = test_data['股票代號'].copy()
    # Drop the column from the DataFrame
    test_data.drop(columns = ['股票代號'], inplace=True)
    test_data = pd.DataFrame(test_scaler.transform(test_data), columns = test_data.columns, index = test_data.index)
    test_data['股票代號'] = stock_number_column

    train_valid_input = []
    train_valid_label = []
    seq_len = 5
    feature_cols = ['low', 'close', 'change', 'open', 'high', 'capacity', 'upper_bb', 'ma5', 'ma20', 'lower_bb', 'dealer_buy', 'dealer_sell', 'investment_buy', 'investment_sell', 'foreign_sell', 'foreign_buy']

    for i in trange(train_data['股票代號'].unique().shape[0]):
        for index in range(train_data.loc[(train_data['股票代號'] == train_data['股票代號'].unique()[i])].shape[0] - seq_len):
            tmp_input = train_data.loc[(train_data['股票代號'] == train_data['股票代號'].unique()[i])][index:index+seq_len+1][feature_cols].reset_index(drop=True)
            input = torch.tensor(np.array(tmp_input.iloc[0:seq_len][feature_cols], dtype = np.float32))
            label = torch.tensor(np.array(tmp_input.iloc[seq_len]['close'], dtype=np.float32))
            train_valid_input.append(input)
            train_valid_label.append(label)

    zeros = torch.zeros(len(train_valid_input), 5, 16)
    for i in range(zeros.shape[0]):
        zeros[i] = train_valid_input[i]
    train_valid_input = zeros
    print('Shape of the training and validating input:', train_valid_input.shape)

    train_valid_label = torch.tensor(train_valid_label)
    print('Shape of the training and validating label', train_valid_label.shape) 

    test_input = []
    test_label = []
    test_stock_number = []

    for i in trange(test_data['股票代號'].unique().shape[0]):
        for index in range(test_data.loc[(test_data['股票代號'] == test_data['股票代號'].unique()[i])].shape[0] - seq_len):
            tmp_input = test_data.loc[(test_data['股票代號'] == test_data['股票代號'].unique()[i])][index:index+seq_len+1][feature_cols].reset_index(drop=True)
            input = torch.tensor(np.array(tmp_input.iloc[0:seq_len][feature_cols], dtype=np.float32))
            label = torch.tensor(np.array(tmp_input.iloc[seq_len]['close'], dtype=np.float32))
            stock_number = torch.tensor(np.array(int(test_data['股票代號'].unique()[i]),dtype=np.float32))
            test_input.append(input)
            test_label.append(label)
            test_stock_number.append(stock_number)

    zeros = torch.zeros(len(test_input), 5, 16)
    for i in range(zeros.shape[0]):
        zeros[i] = test_input[i]
    test_input = zeros
    print('Shape of the testing input:', test_input.shape)

    test_label = torch.tensor(test_label)
    print('Shape of the testing label', test_label.shape)

    test_stock_number = torch.tensor(test_stock_number)
    print('Shape of the testing stock number', test_stock_number.shape)

    train_input = train_valid_input[:int(0.8*len(train_valid_input))]
    train_label = train_valid_label[:int(0.8*len(train_valid_label))]
    valid_input = train_valid_input[int(0.8*len(train_valid_input)):]     
    valid_label = train_valid_label[int(0.8*len(train_valid_label)):] 

    torch.save(train_input, f'dataset/torch-data/train_input{part}.pt')
    torch.save(train_label, f'dataset/torch-data/train_label{part}.pt')
    torch.save(valid_input, f'dataset/torch-data/valid_input{part}.pt')
    torch.save(valid_label, f'dataset/torch-data/valid_label{part}.pt')
    torch.save(test_input, f'dataset/torch-data/test_input{part}.pt')
    torch.save(test_label, f'dataset/torch-data/test_label{part}.pt')
    torch.save(test_stock_number, f'dataset/torch-data/test_stock_number{part}.pt')