import torch

# 加載數據
train_input = torch.load('dataset/torch-data/train_input.pt')
valid_input = torch.load('dataset/torch-data/valid_input.pt')
train_label = torch.load('dataset/torch-data/train_label.pt')
valid_label = torch.load('dataset/torch-data/valid_label.pt')
train_stock_number = torch.load('dataset/torch-data/train_stock_number.pt')
valid_stock_number = torch.load('dataset/torch-data/valid_stock_number.pt')

# 合併數據
train_valid_input = torch.cat((train_input, valid_input), dim=0)
train_valid_label = torch.cat((train_label, valid_label), dim=0)
train_valid_stock_number = torch.cat((train_stock_number, valid_stock_number), dim=0)

# 保存合併後的數據
torch.save(train_valid_input, 'dataset/torch-data/train_valid_input.pt')
torch.save(train_valid_label, 'dataset/torch-data/train_valid_label.pt')
torch.save(train_valid_stock_number, 'dataset/torch-data/train_valid_stock_number.pt')