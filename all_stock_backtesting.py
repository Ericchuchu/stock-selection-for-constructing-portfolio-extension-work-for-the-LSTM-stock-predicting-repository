import yfinance as yf
import talib
import pandas as pd
import numpy as np
import vectorbt as vbt 
from vectorbt.portfolio.enums import OrderSide
import torch
from tqdm import trange


# data preprocess
threshold = 0
all_stock_df = torch.load(f'dataset/torch-data/raw_test_data.pt')
train_valid_stock_number = torch.load(f'dataset/torch-data/train_valid_stock_number.pt').int()
all_stock_df = pd.DataFrame(all_stock_df)
stock_number = []
sharpe_ratio_arr =[]
total_return_arr = []
error_stock = []

for i in range(train_valid_stock_number.unique().shape[0]):   
    target_stock = train_valid_stock_number.unique()[i].numpy()
    # 有些股票在dataset區分時，時間序列資料不連續，會導致回策錯誤，先獨立出來
    try:
        stock_df = all_stock_df[all_stock_df['股票代號'] == target_stock].reset_index(drop = True)
        five_day_min_low = talib.MIN(stock_df['low'], timeperiod=5) # 五天最低價當作long strategy的停損點
        five_day_max_high = talib.MAX(stock_df['high'], timeperiod=5) # 五天最高價當作short strategy的停損點
        stock_df['five_day_min_low'] = five_day_min_low
        stock_df['five_day_max_high'] = five_day_max_high
        stock_df = stock_df[5:]
        predicted_price_df = pd.read_csv(f'predicted_result/{str(target_stock)} 2023 predicted prices.csv', encoding='utf-8', encoding_errors = 'ignore')
        stock_df['日期'] = pd.to_datetime(stock_df['日期'])
        stock_df.index = stock_df['日期']
        stock_df['predicted_price'] = predicted_price_df['predicted_price'].values
        stock_df['true_price'] = predicted_price_df['true_price'].values

        # long short condition
        stock_df['predicted_price_up'] = (stock_df['predicted_price'] > stock_df['predicted_price'].shift(1)).astype(int)
        stock_df['actual_price_up'] = (stock_df['close'] > stock_df['close'].shift(1)).astype(int)
        stock_df['long_condition'] = (stock_df['true_price'].shift(1)<=stock_df['predicted_price']) & ((stock_df['predicted_price']-stock_df['true_price'].shift(1))/stock_df['true_price'].shift(1) >= threshold)
        stock_df['short_condition'] = (stock_df['true_price'].shift(1)>stock_df['predicted_price']) & (abs((stock_df['predicted_price']-stock_df['true_price'].shift(1))/stock_df['true_price'].shift(1)) >= threshold)
        stock_df['long_condition'] = stock_df['long_condition'].astype(int)
        stock_df['short_condition'] = stock_df['short_condition'].astype(int)


        # long strategy
        signals_long = np.zeros(len(stock_df))
        partition_size_long = 0
        stop_loss_long = 0
        for i in range(len(stock_df)):
            if partition_size_long == 0:
                if stock_df['long_condition'].iloc[i] == 1:
                    signals_long[i] = 1
                    partition_size_long = 1
                    stop_loss_long = stock_df['five_day_min_low'].iloc[i]
            if partition_size_long == 1:
                if (stock_df['close'].iloc[i] <= stop_loss_long) or (stock_df['short_condition'].iloc[i] == 1): # 停利停損
                    signals_long[i] = -1
                    stop_loss_long = 0 # 停損點重置
                    partition_size_long = 0 # 平倉

        # short strategy
        signals_short = np.zeros(len(stock_df))
        partition_size_short = 0
        stop_loss_short = 0
        for i in range(len(stock_df)):
            if partition_size_short == 0:
                if stock_df['short_condition'].iloc[i] == 1:
                    signals_short[i] = -1
                    partition_size_short = -1
                    stop_loss_short = stock_df['five_day_max_high'].iloc[i]
            if partition_size_short == -1:
                if (stock_df['close'].iloc[i] >= stop_loss_short) or (stock_df['long_condition'].iloc[i] == 1):# 停利停損
                    signals_short[i] = 1
                    stop_loss_short = 0 # 停損點重置
                    partition_size_short = 0 # 平倉

        entries_long = signals_long == 1
        exits_long = signals_long == -1
        entries_short = signals_short == -1
        exits_short = signals_short == 1
        
        pf = vbt.Portfolio.from_signals(stock_df['open'], 
                                            entries=entries_long,
                                            exits=exits_long, 
                                            short_entries=entries_short,
                                            short_exits=exits_short,
                                            fees = 0,
                                            freq='1D',)

        sharpe_ratio = float(pf.sharpe_ratio())
        total_return = float(pf.total_return())
        
        stock_number.append(target_stock)
        sharpe_ratio_arr.append(sharpe_ratio)
        total_return_arr.append(total_return)

        print(str(target_stock))
        print("Sharpe Ratio:", sharpe_ratio)
        print("Total Return:", total_return)
    except:
        error_stock.append(target_stock) 
        
        
result_df = pd.DataFrame({'stock_number':stock_number, 'sharpe_ratio':sharpe_ratio_arr, 'total_return': total_return_arr})
result_df.to_csv(f'backtesting_result/sharpe_and_return.csv')
print(error_stock)#3401 3202