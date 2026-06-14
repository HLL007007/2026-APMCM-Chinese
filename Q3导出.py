import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Add, Lambda
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ==========================================
# 0. 环境与画图设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取数据与严谨的因果清洗
# ==========================================
file_path = "Combined_Water_Quality_2025_2026_Q3.xlsx" 
print(f"正在读取并清洗数据: {file_path} ...")
df = pd.read_excel(file_path)

df.columns = [str(col).strip().replace('\n', '') for col in df.columns]
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['DATE_STR'] = df['DATE'].dt.strftime('%Y-%m-%d')
df['TIME_STR'] = df['TIME'].astype(str).str.replace(':', '').str.replace('.0', '', regex=False).str.strip().str.zfill(4)

target_col = 'NTU' 
state_features = ['FILT. NTU', 'R/W FLOW', 'C/W WELL LEVEL', 'T/W FLOW']
control_features = ['F/RIDE', 'T/W PUMP DUTY']

# 强制转换并填充
for col in state_features + control_features + [target_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].ffill().bfill() 

# 引入时滞 (滞后2步/4小时)
df['F/RIDE_Lag2'] = df['F/RIDE'].shift(2).ffill().bfill()
features = state_features + ['F/RIDE_Lag2', 'T/W PUMP DUTY']

# ==========================================
# 2. 物理先验特征
# ==========================================
print("正在计算物理守恒边界...")
epsilon = 1e-5
df['HRT'] = df['C/W WELL LEVEL'] / (df['T/W FLOW'] + epsilon)
df['Phys_Delta_NTU'] = (df['R/W FLOW'] * df['FILT. NTU'] - df['T/W FLOW'] * df[target_col].shift(1).bfill()) / (df['C/W WELL LEVEL'] + epsilon)

all_features = features + ['HRT', 'Phys_Delta_NTU']

# ==========================================
# 3. 构建 Seq2Seq 张量 (修改 horizon 为 12)
# ==========================================
print("构建时空张量与特征缩放...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

df_scaled_features = scaler_X.fit_transform(df[all_features])
data_y_scaled = scaler_y.fit_transform(df[[target_col]])

lookback = 12 # 过去 24 小时
horizon = 12  # 【修改处】预测未来 12 步 (涵盖 07:00 到次日 05:00)

X_seq, Y_seq = [], []
for i in range(lookback, len(df) - horizon):
    X_seq.append(df_scaled_features[i-lookback : i, :])
    Y_seq.append(data_y_scaled[i : i+horizon, 0]) 

X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)

split_idx = int(len(X_seq) * 0.8)
X_train, y_train = X_seq[:split_idx], Y_seq[:split_idx]
X_test, y_test = X_seq[split_idx:], Y_seq[split_idx:]

# ==========================================
# 4. 训练 PINN-LSTM 
# ==========================================
print("\n构建并训练带有物理惩罚项的 PINN-LSTM (预测周期扩展为全天 24H)...")

def pinn_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    physics_penalty = tf.reduce_mean(tf.square(pred_diff)) 
    return mse_loss + 0.1 * physics_penalty

inputs = Input(shape=(lookback, len(all_features)))
lstm_out = LSTM(64, return_sequences=True)(inputs)
lstm_out = LSTM(32)(lstm_out)
dense_out = Dense(32, activation='relu')(lstm_out)
final_out = Dense(horizon, name='prediction')(dense_out) # 自动适配 12 步

model = Model(inputs=inputs, outputs=final_out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=pinn_loss)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)

# ==========================================
# 5. 生成 2026年2月 完整预测表 (严格按图示格式输出)
# ==========================================
print("\n" + "="*50)
print("开始执行 2026 年 2 月全月逐小时预测...")

# 获取二月的所有日期 (2026年不是闰年，2月有28天)
feb_dates = pd.date_range(start='2026-02-01', end='2026-02-28').strftime('%Y-%m-%d')
# 严格按照图中的时间顺序 (注意：0100, 0300, 0500 属于业务上的同一天)
display_times = ['0700', '0900', '1100', '1300', '1500', '1700', 
                 '1900', '2100', '2300', '0100', '0300', '0500']

results = []

for t_date in feb_dates:
    # 每天以 0700 为起点进行预测
    mask = (df['DATE_STR'] == t_date) & (df['TIME_STR'] == '0700')
    if mask.any():
        start_idx = df[mask].index[0] 
        if start_idx >= lookback:
            # 获取过去 12 个步长的数据作为预测输入
            input_tensor = np.array([df_scaled_features[start_idx-lookback : start_idx]])
            pred_scaled = model.predict(input_tensor, verbose=0)
            # 反归一化得到真实 NTU 值
            pred_real = scaler_y.inverse_transform(pred_scaled).flatten()
            
            # 格式化填入列表中
            for time_str, val in zip(display_times, pred_real):
                results.append({
                    'DATE': f"{t_date} 00:00:00",  # 完全匹配图中的 DATE 格式
                    'TIME': time_str,              # 完全匹配图中的 TIME 格式
                    'NTU': max(round(val, 4), 0.05) # 限制最低预测值为 0.05
                })

# 转换为 DataFrame 并导出
ans_df = pd.DataFrame(results)
excel_filename = 'Feb_2026_NTU_Full_Predictions.xlsx'
ans_df.to_excel(excel_filename, index=False)

print(f"✅ 预测完成！已成功导出全月数据至: {excel_filename}")
print(ans_df.head(15)) # 打印前15行预览效果