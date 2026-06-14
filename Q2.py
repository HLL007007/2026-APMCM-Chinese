import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # 引入随机切分
from statsmodels.tsa.stattools import ccf

# 导入 TensorFlow / Keras 构建高级网络
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ==========================================
# 0. 自定义函数与环境设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows系统防乱码，Mac请换用 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

# 定义 Hampel 滤波器函数 (基于 Median Absolute Deviation)
def hampel_filter(series, window_size=5, n_sigmas=3):
    rolling_median = series.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = series.rolling(window=window_size, center=True, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))))
    threshold = n_sigmas * 1.4826 * rolling_mad
    outlier_mask = np.abs(series - rolling_median) > threshold
    series_cleaned = series.copy()
    series_cleaned[outlier_mask] = np.nan
    return series_cleaned

# ==========================================
# 1. 数据读取与日期解析清洗
# ==========================================
file_path = "Combined_Water_Quality_2025_2026_Q2.xlsx" 
print(f"正在读取数据: {file_path} ...")
# 替换为你的真实路径和读取方式
df = pd.read_excel(file_path)
df.columns = [str(col).strip() for col in df.columns]

df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

target_col = 'FILT. NTU'
features = ['R/W FLOW', 'R/W NTU', 'R/W PH', 'ALUM']

for col in features + [target_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['DATE', target_col])

print("\n[数据预处理] 执行物理约束、Hampel 滤波与插值...")

for col in features + [target_col]:
    df.loc[df[col] < 0, col] = np.nan
df.loc[df[target_col] > 2.0, target_col] = np.nan

for col in features + [target_col]:
    df[col] = hampel_filter(df[col], window_size=5, n_sigmas=3)
    df[col] = df[col].rolling(window=3, center=True, min_periods=1).median()

df[features + [target_col]] = df[features + [target_col]].interpolate(method='linear').ffill().bfill()


# ==========================================
# 2. 互相关函数 (CCF) 计算寻找物理时滞
# ==========================================
print("\n正在计算 CCF 寻找最佳时滞，以为 TCN 确定感受野窗口大小...")
max_lag = 12 
lag_dict = {}

fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("各特征与出厂水浊度的互相关函数 (CCF)", fontsize=18, fontweight='bold', y=0.95)
axes = axes.flatten()

for i, feature in enumerate(features):
    cross_corr = ccf(df[target_col], df[feature], adjusted=False)[:max_lag+1]
    best_lag = np.argmax(np.abs(cross_corr[1:])) + 1 
    lag_dict[feature] = best_lag
    
    ax = axes[i]
    ax.stem(range(max_lag + 1), cross_corr, basefmt=" ")
    ax.axvline(x=best_lag, color='red', linestyle='--', linewidth=2, label=f'最大影响时滞: {best_lag}')
    ax.set_title(f"[{feature}] 的时滞特征", fontsize=14)
    ax.set_xlabel("滞后步数 (Lags)", fontsize=12)
    ax.set_ylabel("相关系数", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend()
plt.tight_layout()


# ==========================================
# 3. 构造深度学习 TCN-Attention 专用的 3D 张量数据
# ==========================================
print("\n正在构建 3D 时间窗口序列数据...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

input_cols = features + [target_col]
data_scaled = scaler_X.fit_transform(df[input_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

seq_len = max_lag 

X_seq, Y_seq = [], []
for i in range(seq_len, len(data_scaled)):
    X_seq.append(data_scaled[i-seq_len : i, :]) 
    Y_seq.append(y_scaled[i, 0])                

X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)


# ==========================================
# 4. ★按 7:3 随机划分训练集与测试集 (带时间序列排序保护)★
# ==========================================
print("\n[模型验证策略] 正在按 70% 训练集, 30% 测试集进行随机切分...")

# 提取索引序列进行随机切分，保留索引以便后续画折线图时排序
indices = np.arange(len(X_seq))

idx_train, idx_test = train_test_split(indices, test_size=0.3, random_state=42)

# ★核心保护机制★：对测试集的索引进行升序排序，保证后续画折线图时是连续的波形
idx_test_sorted = np.sort(idx_test)

X_train, y_train = X_seq[idx_train], Y_seq[idx_train]
X_test, y_test = X_seq[idx_test_sorted], Y_seq[idx_test_sorted]

print(f"-> 训练集样本数: {len(X_train)} 条")
print(f"-> 测试集样本数: {len(X_test)} 条 (已恢复时间顺序)")


# ==========================================
# 5. ★构建核武器级架构: TCN-Attention 模型★
# ==========================================
print("\n正在构建高级 TCN-Attention 深度网络 (请耐心等待数十秒)...")

# 使用 Functional API 构建网络
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# 第一阶段：TCN 的因果膨胀卷积 (捕获局部时滞与突变)
x = Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu', dilation_rate=1)(inputs)
x = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu', dilation_rate=2)(x)
tcn_out = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu', dilation_rate=4)(x)

# 第二阶段：多头自注意力机制 (Multi-Head Self-Attention) (捕获全局周期性依赖)
attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(tcn_out, tcn_out)

# 加入残差连接与层归一化 (防止梯度消失，加速收敛)
x = Add()([tcn_out, attn_out])
x = LayerNormalization()(x)

# 第三阶段：展平与回归输出
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# 监控训练集的误差，自动早停防止过拟合
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1, # 从训练集中划分 10% 供早停验证
    callbacks=[early_stop],
    verbose=0 
)

# 预测与逆标准化
y_train_pred_scaled = model.predict(X_train, verbose=0)
y_test_pred_scaled = model.predict(X_test, verbose=0)

y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
y_train_real = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()


# ==========================================
# 6. 计算测试集的四大检验参数
# ==========================================
r2 = r2_score(y_test_real, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test_real, y_test_pred))
mae = mean_absolute_error(y_test_real, y_test_pred)
mape = mean_absolute_percentage_error(y_test_real, y_test_pred) * 100

print("\n" + "="*60)
print("【TCN-Attention 模型】在 30% 测试集上的终极验证精度")
print("="*60)
print(f"决定系数 (R²) : {r2:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE) : {mae:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")


# ==========================================
# 7. 终极可视化：生成分离的高清学术图表
# ==========================================
print("\n正在渲染所有高清学术图表...")

# 【生成图 2】：测试集时序动态拟合曲线
fig2 = plt.figure(figsize=(14, 6))
plt.plot(y_test_real, label='真实浊度 (测试集片段)', color='#1f77b4', alpha=0.8, linewidth=1.5)
plt.plot(y_test_pred, label='TCN-Attention 预测值', color='#ff7f0e', alpha=0.9, linestyle='--', linewidth=1.8)
plt.title(f"TCN-Attention 测试集 (30%片段) 动态拟合曲线 (R²={r2:.4f})", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("测试集样本点 (已还原时序)", fontsize=13)
plt.ylabel("滤后水浊度 (NTU)", fontsize=13)
plt.legend(fontsize=12, loc='upper left')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()

# 【生成图 3】：★训练集与测试集分离的 双散点图★
fig3, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图 3-1: 训练集散点图
ax1 = axes[0]
ax1.scatter(y_train_real, y_train_pred, color='#3498db', alpha=0.5, edgecolor='w', s=50)
min_train = min(y_train_real.min(), y_train_pred.min())
max_train = max(y_train_real.max(), y_train_pred.max())
ax1.plot([min_train, max_train], [min_train, max_train], 'r--', linewidth=2.5, label='完美拟合线 (y=x)')
r2_train = r2_score(y_train_real, y_train_pred)
ax1.set_title(f"【训练集 (70% 样本)】 散点分布 ($R^2$: {r2_train:.4f})", fontsize=14, fontweight='bold')
ax1.set_xlabel("真实值 (Actual NTU)", fontsize=12)
ax1.set_ylabel("注意力模型预测值", fontsize=12)
ax1.legend()
ax1.grid(linestyle='--', alpha=0.5)

# 子图 3-2: 测试集散点图
ax2 = axes[1]
ax2.scatter(y_test_real, y_test_pred, color='#2ca02c', alpha=0.6, edgecolor='w', s=50)
min_test = min(y_test_real.min(), y_test_pred.min())
max_test = max(y_test_real.max(), y_test_pred.max())
ax2.plot([min_test, max_test], [min_test, max_test], 'r--', linewidth=2.5, label='完美拟合线 (y=x)')
ax2.set_title(f"【测试集 (30% 样本)】 散点分布 ($R^2$: {r2:.4f})", fontsize=14, fontweight='bold')
ax2.set_xlabel("真实值 (Actual NTU)", fontsize=12)
ax2.set_ylabel("注意力模型预测值", fontsize=12)
ax2.legend()
ax2.grid(linestyle='--', alpha=0.5)

plt.tight_layout()

# 【生成图 4】：三大检验参数柱状图 (测试集)
fig4 = plt.figure(figsize=(10, 6))
metrics_names = ['决定系数\n$R^2$', '均方根误差\nRMSE', '平均绝对误差\nMAE']
metrics_values = [r2, rmse, mae]
colors = ['#4CB391', '#e74c3c', '#9b59b6']

bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black', alpha=0.85, width=0.5)

for i, bar in enumerate(bars):
    yval = bar.get_height()
    label = f'{yval:.4f}' if i < 3 else f'{yval:.2f}%'
    offset = max(metrics_values) * 0.03 
    plt.text(bar.get_x() + bar.get_width()/2, yval + offset, label, 
             ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.title("TCN-Attention 模型核心评估指标总览", fontsize=16, fontweight='bold', pad=15)
plt.ylim(0, max(metrics_values) * 1.25) 
plt.ylabel("指标统计数值", fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

print("\n✅ TCN-Attention (7:3 随机划分) 训练验证执行完毕！请查看弹出的 4 个独立窗口。")