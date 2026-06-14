import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor 
from pygam import LinearGAM, s
import warnings

warnings.filterwarnings('ignore') 

# ========================================================
# 0. 环境与画图设置
# ========================================================
# 设置中文字体，防止画图时中文乱码 (Mac系统请改为 'Arial Unicode MS')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ========================================================
# 1. 核心预处理与特征工程函数定义 (强力加分项)
# ========================================================
def calc_pump_count(x):
    """
    【特征工程】将离散文本状态转化为物理意义上的“开启台数”
    例如: '1,2' -> 2; '1+3' -> 2; '1' -> 1; 缺失 -> 0
    """
    if pd.isna(x) or str(x).strip() in ['', '-']:
        return 0
    # 统一替换非法符号为逗号，并去除空格
    s = str(x).replace('+', ',').replace('&', ',').replace(' ', '')
    # 按照逗号分割，计算有多少个独立的水泵编号
    return len([p for p in s.split(',') if p])

def robust_clean_data(df):
    """
    【数据清洗】对工业水质数据进行专业级的异常值截断、去噪平滑与缺失值插补
    """
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # 1. 移动中值滤波去噪 (窗口=3)
        df_cleaned[col] = df_cleaned[col].rolling(window=3, center=True, min_periods=1).median()
        
        # 2. 极端异常值盖帽截断 (1% ~ 99%)
        lower_bound = df_cleaned[col].quantile(0.01)
        upper_bound = df_cleaned[col].quantile(0.99)
        df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        
    # 3. 线性插值与前后向填充
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].interpolate(method='linear').ffill().bfill()
    return df_cleaned


# ========================================================
# 第一部分：特征重要性分析 (皮尔逊、XGBoost、SHAP)
# ========================================================
print("\n" + "="*50)
print(" 第一部分：特征重要性与归因分析")
print("="*50)

file_path = "Combined_Water_Quality_2025_2026_Q1.xlsx"
print(f"正在读取历史训练数据: {file_path} ...")
df_raw = pd.read_excel(file_path)

# 清理表头首尾空格
df_raw.columns = [str(col).strip() for col in df_raw.columns]

# --- 应用特征工程：转化泵机状态 ---
if 'T/W PUMP DUTY' in df_raw.columns:
    df_raw['T/W PUMP DUTY'] = df_raw['T/W PUMP DUTY'].apply(calc_pump_count)
if 'R/W PUMP DUTY' in df_raw.columns:
    df_raw['R/W PUMP DUTY'] = df_raw['R/W PUMP DUTY'].apply(calc_pump_count)

# 锁定目标变量 (L列为 NTU 出厂水浊度)
target_col = df_raw.columns[11] 
print(f"[锁定目标] 提取预测目标: '{target_col}'")

# 剔除日期和时间列进行数值分析
date_time_keywords = ['date', 'time', '日期', '时间']
drop_cols = [col for col in df_raw.columns if any(kw in col.lower() for kw in date_time_keywords)]
df_numeric = df_raw.drop(columns=drop_cols, errors='ignore')

# 强制转换数值型
df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

# 执行专业级的数据清洗 (去噪、截断、插补)
print("\n[数据预处理] 正在执行 缺失值插补、去噪与异常值截断...")
df_numeric = robust_clean_data(df_numeric)

# 分离特征 X 和 标签 y
X_eda = df_numeric.drop(columns=[target_col], errors='ignore')
y_eda = df_numeric[target_col]

# 1. 生成图表：皮尔逊相关性热力图
print("\n正在生成 皮尔逊相关性热力图...")
plt.figure(figsize=(16, 12)) 
sns.heatmap(df_numeric.corr(), annot=True, annot_kws={"size": 10}, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("变量间皮尔逊相关性热力图", fontsize=18, pad=20)
plt.tight_layout()
plt.show()

# 2. XGBoost 模型训练与特征重要性排名
print("\n正在训练 XGBoost 模型寻找核心变量...")
xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
xgb_model.fit(X_eda, y_eda)

feature_importances = pd.Series(xgb_model.feature_importances_, index=X_eda.columns).sort_values(ascending=True)
plt.figure(figsize=(10, 8))
feature_importances.plot(kind='barh', color='#4CB391', edgecolor='black')
plt.title(f"XGBoost 核心影响因素排名 (目标: {target_col})", fontsize=16)
plt.xlabel("特征重要性权重 (Feature Importance)", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. SHAP 归因分析图
print("\n正在生成 SHAP 归因分析图...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_eda)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_eda, plot_type="dot", show=False)
plt.title(f"{target_col} 的 SHAP 全局归因分析", fontsize=16)
plt.tight_layout()
plt.show()


# ========================================================
# 第二部分：三大核心模型预测 (Lasso、GAM、Random Forest)
# ========================================================
print("\n" + "="*50)
print(" 第二部分：模型建立、对比与未知数据预测")
print("="*50)

predict_file = "2026-02.xlsx"
print(f"正在读取待预测集数据: {predict_file} ...")
df_predict_raw = pd.read_excel(predict_file)
df_predict_raw.columns = [str(col).strip() for col in df_predict_raw.columns]

# --- 应用特征工程：预测集泵机转化 ---
if 'T/W PUMP DUTY' in df_predict_raw.columns:
    df_predict_raw['T/W PUMP DUTY'] = df_predict_raw['T/W PUMP DUTY'].apply(calc_pump_count)
if 'R/W PUMP DUTY' in df_predict_raw.columns:
    df_predict_raw['R/W PUMP DUTY'] = df_predict_raw['R/W PUMP DUTY'].apply(calc_pump_count)

# 锁定前五大核心特征 (此时 T/W PUMP DUTY 已经是安全的纯数字了)
top5_features = ['FILT. NTU', 'F/RIDE', 'T/W PUMP DUTY' , 'T/W FLOW' , 'R/W FLOW']
print(f"\n[锁定模型输入特征] {top5_features}")

# 提取并构建模型训练集
X_model = df_numeric[top5_features]
y_model = df_numeric[target_col]

# 提取预测集特征，强转数值，并应用去噪清洗流程
time_col = [col for col in df_predict_raw.columns if any(kw in col.lower() for kw in ['date', 'time', '日期', '时间'])]
time_col = time_col[0] if time_col else None

df_predict_numeric = df_predict_raw[top5_features].apply(pd.to_numeric, errors='coerce')
df_predict_cleaned = robust_clean_data(df_predict_numeric) 

# ★ 终极兜底防线：防止预测集中由于全为空值导致的 NaN 传递
df_predict_cleaned = df_predict_cleaned.fillna(0)

# 特征缩放与数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_predict_scaled = scaler.transform(df_predict_cleaned)

eval_metrics = []

# 初始化导出 Excel 的基础结构
base_output_df = df_predict_raw[[time_col]].copy() if time_col else pd.DataFrame()
for col in top5_features:
    base_output_df[col] = df_predict_cleaned[col]


# --- 模型 1：Lasso 回归 ---
print("\n[1] 正在训练 Lasso 回归模型...")
lasso = Lasso(alpha=0.001)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

eval_metrics.append({
    'Model': 'Lasso Regression',
    'R2': r2_score(y_test, y_pred_lasso),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    'MAE': mean_absolute_error(y_test, y_pred_lasso)
})

df_lasso = base_output_df.copy()
df_lasso['Lasso_Pred_NTU'] = lasso.predict(X_predict_scaled)

formula_lasso = f"Y = {lasso.intercept_:.4f}"
for i, feature in enumerate(top5_features):
    if lasso.coef_[i] != 0:
        sign = "+" if lasso.coef_[i] > 0 else "-"
        formula_lasso += f" {sign} {abs(lasso.coef_[i]):.4f} * {feature}"
print("-> Lasso 函数关系式:\n", formula_lasso)


# --- 模型 2：GAM 广义加性模型 ---
print("\n[2] 正在训练 GAM 模型...")
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4))
gam.gridsearch(X_train.values, y_train.values, progress=False)
y_pred_gam = gam.predict(X_test.values)

eval_metrics.append({
    'Model': 'GAM (广义加性模型)',
    'R2': r2_score(y_test, y_pred_gam),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gam)),
    'MAE': mean_absolute_error(y_test, y_pred_gam)
})

df_gam = base_output_df.copy()
df_gam['GAM_Pred_NTU'] = gam.predict(df_predict_cleaned.values)

print(f"-> GAM 基准截距 (β0): {gam.coef_[-1]:.4f}")
print("-> 正在生成 GAM 函数形态偏相关图...")
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('GAM 各变量偏相关图', fontsize=16, y=0.95)

for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    pdp, conf = gam.partial_dependence(term=i, X=XX, width=0.95)
    ax.plot(XX[:, i], pdp, color='#1f77b4', linewidth=2)
    ax.plot(XX[:, i], conf[:, 0], color='red', linestyle='--', alpha=0.5)
    ax.plot(XX[:, i], conf[:, 1], color='red', linestyle='--', alpha=0.5)
    ax.set_title(f"f_{i+1}({top5_features[i]})", fontsize=12)
    ax.set_xlabel(top5_features[i])
    if i == 0: ax.set_ylabel("对出厂水浊度的偏影响")
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# --- 模型 3：Random Forest 随机森林 ---
print("\n[3] 正在训练 Random Forest 模型...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

eval_metrics.append({
    'Model': 'Random Forest',
    'R2': r2_score(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'MAE': mean_absolute_error(y_test, y_pred_rf)
})

df_rf = base_output_df.copy()
df_rf['RF_Pred_NTU'] = rf_model.predict(X_predict_scaled)


# ========================================================
# 结果汇总与导出
# ========================================================
df_eval = pd.DataFrame(eval_metrics)
print("\n" + "="*50)
print(" 模型评价指标对比汇总")
print("="*50)
print(df_eval.to_string(index=False))

file_lasso = 'Predict_Result_Lasso_Feb2026.xlsx'
file_gam = 'Predict_Result_GAM_Feb2026.xlsx'
file_rf = 'Predict_Result_RandomForest_Feb2026.xlsx'

df_lasso.to_excel(file_lasso, index=False)
df_gam.to_excel(file_gam, index=False)
df_rf.to_excel(file_rf, index=False)
print(f"\n[导出成功] 三个模型的预测结果已分别保存为:")
print(f"1. {file_lasso}\n2. {file_gam}\n3. {file_rf}")

# 模型效果对比三联图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#4CB391', '#FF9F43', '#00CFE8']

axes[0].bar(df_eval['Model'], df_eval['R2'], color=colors, edgecolor='black')
axes[0].set_title("各模型决定系数 ($R^2$) 对比 - 越高越好", fontsize=14)
axes[0].set_ylim(0, 1) 
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

axes[1].bar(df_eval['Model'], df_eval['RMSE'], color=colors, edgecolor='black')
axes[1].set_title("各模型均方根误差 (RMSE) 对比 - 越低越好", fontsize=14)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

axes[2].bar(df_eval['Model'], df_eval['MAE'], color=colors, edgecolor='black')
axes[2].set_title("各模型平均绝对误差 (MAE) 对比 - 越低越好", fontsize=14)
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()