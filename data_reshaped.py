import pandas as pd

# Excel文件路径
excel_file = f'dataset/dataset.xlsx'
# 初始劃一個空的dataframe來合併數據
merged_df = pd.DataFrame()
with pd.ExcelFile(excel_file) as xls:
    # 遍歷所有工作表
    for sheet_name in xls.sheet_names:
        # 讀取當前工作表
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df_long = pd.melt(df, id_vars=['股票代號', '股票名稱'], var_name='日期', value_name=sheet_name)
        df_long['日期'] = df_long['日期'].str[:8]
        df_long['日期'] = pd.to_datetime(df_long['日期'], format='%Y%m%d')
        df_sorted = df_long.sort_values(by=['股票代號', '日期']).reset_index(drop = True)
        # 合併到大的DataFrame
        if merged_df.empty:
            merged_df = df_sorted
        else:
            merged_df = pd.merge(merged_df, df_sorted, on=['股票代號', '股票名稱', '日期'], how='outer')
        
# 丟掉缺失值
merged_df = merged_df.dropna()    

# 分割dataframe
max_rows = 100000
for i, start in enumerate(range(0, len(merged_df), max_rows)):
    df_part = merged_df.iloc[start:start + max_rows]
    with pd.ExcelWriter(f'preprocessed_data/preprocessed_data_part{i}.xlsx') as writer:
        df_part.to_excel(writer, index=False, sheet_name=f'Sheet{i}')