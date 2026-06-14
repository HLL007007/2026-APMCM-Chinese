import pandas as pd
import os
import glob
import datetime
import numpy as np

# 定义数据文件夹路径
base_dir = 'data'
dir_2025 = os.path.join(base_dir, '2025')
dir_2026 = os.path.join(base_dir, '2026')

all_data_frames = []

# 月份映射表
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def get_true_month(filename):
    filename_lower = filename.lower()
    for month_str, month_num in month_map.items():
        if month_str in filename_lower:
            return month_num
    return 1

def fix_2025_date(val, true_year, true_month):
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
        if val.day == true_month:
            true_day = val.month
        elif val.month == true_month:
            true_day = val.day
        else:
            true_day = val.day
        return pd.Timestamp(year=true_year, month=true_month, day=true_day)

    if isinstance(val, str):
        try:
            parsed_date = pd.to_datetime(val, dayfirst=True)
            return pd.Timestamp(year=true_year, month=true_month, day=parsed_date.day)
        except:
            pass
    return pd.NaT

print("开始处理 2025 年数据...")
for file_path in glob.glob(os.path.join(dir_2025, '*.xlsx')):
    try:
        filename = os.path.basename(file_path)
        true_month = get_true_month(filename)
        
        df = pd.read_excel(file_path)
        
        df.columns = df.columns.astype(str).str.replace('\n', '').str.replace('\r', '').str.strip().str.upper()
        
        rename_mapping = {}
        for col in df.columns:
            if 'DAT' in col:
                rename_mapping[col] = 'DATE'
            elif 'TIM' in col:
                rename_mapping[col] = 'TIME'
        df = df.rename(columns=rename_mapping)
    
        df = df.dropna(subset=['TIME'])
        
        
        df['DATE'] = df['DATE'].ffill()
        
        df['DATE'] = df['DATE'].apply(lambda x: fix_2025_date(x, 2025, true_month))
        
        df = df.dropna(subset=['DATE'])
        
        all_data_frames.append(df)
        print(f"成功读取并修复: {filename}")
    except Exception as e:
        print(f"读取 {file_path} 时出错: {e}")

print("\n开始处理 2026 年数据...")
for file_path in glob.glob(os.path.join(dir_2026, '*.xls*')):
    try:
        year = 2026 
        sheets = pd.read_excel(file_path, sheet_name=None)
        
        for sheet_name, df in sheets.items():
            if '.' not in sheet_name:
                continue
                
            df.columns = df.columns.astype(str).str.replace('\n', '').str.replace('\r', '').str.strip().str.upper()
            
            # 同样应用容错重命名（防止2026年也有问题）
            rename_mapping = {}
            for col in df.columns:
                if 'TIM' in col:
                    rename_mapping[col] = 'TIME'
            df = df.rename(columns=rename_mapping)
            
            df = df.dropna(subset=['TIME'])
            
            if df.empty:
                continue
                
            try:
                day, month = sheet_name.split('.')
                date_str = f"{year}-{int(month):02d}-{int(day):02d}"
            except ValueError:
                continue
                
            df.insert(0, 'DATE', pd.to_datetime(date_str))
            all_data_frames.append(df)
            
        print(f"成功读取: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"读取 {file_path} 时出错: {e}")

print("\n正在执行最终清洗与合并...")
if all_data_frames:
    final_df = pd.concat(all_data_frames, ignore_index=True)
    
    # 清洗 TIME 列
    final_df['TIME'] = final_df['TIME'].astype(str).str.replace(r'\.0$', '', regex=True).str.replace("'", "").str.zfill(4)

    # 清洗泵机状态
    pump_cols = ['R/W PUMP DUTY', 'T/W PUMP DUTY']
    for col in pump_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna('').astype(str)
            final_df[col] = final_df[col].str.replace(r'[+&]', ',', regex=True)
            final_df[col] = final_df[col].str.replace(r'\s+', '', regex=True)
            final_df[col] = final_df[col].replace('', np.nan)

    shift_order = {
        '0700': 1, '0900': 2, '1100': 3, '1300': 4,
        '1500': 5, '1700': 6, '1900': 7, '2100': 8,
        '2300': 9, '0100': 10, '0300': 11, '0500': 12
    }
    
    final_df['SHIFT_ORDER'] = final_df['TIME'].map(shift_order).fillna(99)
    final_df = final_df.sort_values(by=['DATE', 'SHIFT_ORDER', 'TIME']).reset_index(drop=True)
    final_df = final_df.drop(columns=['SHIFT_ORDER'])
    
    output_filename = 'Combined_Water_Quality_2025_2026.xlsx'
    final_df.to_excel(output_filename, index=False)
    
    print(f"\n共合并 {len(final_df)} 条数据。")
    print(f"结果已保存为: {output_filename}")
else:
    print("未能找到任何数据，请检查文件夹路径。")