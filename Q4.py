import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 环境与画图设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows系统防乱码
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据读取与 1、2、3 月数据截取
# ==========================================
file_path = "Combined_Water_Quality_2025_2026_Q4.xlsx" 
print(f"正在读取数据: {file_path} ...")
df_all = pd.read_excel(file_path)

df_all.columns = [str(col).strip().replace('\n', '') for col in df_all.columns]
df_all['DATE'] = pd.to_datetime(df_all['DATE'], errors='coerce')

target_col = 'NTU' 
df_all[target_col] = pd.to_numeric(df_all[target_col], errors='coerce')
df_all = df_all.dropna(subset=['DATE', target_col])

# ★ 截取 2026年 1, 2, 3 月的数据 ★
mask_q1 = (df_all['DATE'].dt.year == 2026) & (df_all['DATE'].dt.month.isin([1, 2, 3]))
df_eval = df_all[mask_q1].copy()
print(f"成功截取 2026年 Q1 水质数据，共计 {len(df_eval)} 条记录。")


# ==========================================
# 2. 日级特征工程与“一票否决”安全判定
# ==========================================
print("\n正在执行日级特征聚合与国标 (≤ 1 NTU) 判定...")
daily_records = []

for date, group in df_eval.groupby(df_eval['DATE'].dt.date):
    ntu_values = group[target_col].values
    max_ntu = np.nanmax(ntu_values)
    month = pd.to_datetime(date).month
    
    # 划分数据集阵营
    dataset_type = 'Train (1,3月)' if month in [1, 3] else 'Test (2月)'
    
    if max_ntu <= 1.0:
        daily_records.append({
            'Date': date, 'Month': month, 'Dataset': dataset_type,
            'Status': '安全 (Safe)', 'Risk_Level': 0, 'Risk_Score': 0.0,
            'M_max': 0, 'M_mean': 0, 'D_total': 0, 'D_cont': 0
        })
    else:
        exceed_mask = ntu_values > 1.0
        exceed_vals = ntu_values[exceed_mask]
        
        m_max = max_ntu - 1.0
        m_mean = np.mean(exceed_vals - 1.0)
        d_total = np.sum(exceed_mask) * 2 
        
        max_cont, current_cont = 0, 0
        for is_exceed in exceed_mask:
            if is_exceed:
                current_cont += 1
                max_cont = max(max_cont, current_cont)
            else:
                current_cont = 0
        d_cont = max_cont * 2
        
        daily_records.append({
            'Date': date, 'Month': month, 'Dataset': dataset_type,
            'Status': '异常待评 (Unsafe)', 'Risk_Level': None, 'Risk_Score': None,
            'M_max': m_max, 'M_mean': m_mean, 'D_total': d_total, 'D_cont': d_cont
        })

df_daily = pd.DataFrame(daily_records)
unsafe_df = df_daily[df_daily['Status'] == '异常待评 (Unsafe)'].copy()
print(f"初步判定：安全 {len(df_daily[df_daily['Status'] == '安全 (Safe)'])} 天，超标异常 {len(unsafe_df)} 天。")


# ==========================================
# 3. ★ 核心分离：基于训练集 (1/3月) 的 AHP-EWM 赋权 ★
# ==========================================
feature_names = ['最大超标\n幅度 (M_max)', '平均超标\n幅度 (M_mean)', '日总超标\n时长 (D_total)', '最大连续\n时长 (D_cont)']
features = ['M_max', 'M_mean', 'D_total', 'D_cont']

if len(unsafe_df) > 0:
    print("\n正在启动 样本外验证版 AHP-EWM 联合赋权模型...")
    
    # 强制分离训练集(1,3月)和测试集(2月)
    train_mask = unsafe_df['Month'].isin([1, 3])
    test_mask = unsafe_df['Month'] == 2
    
    X_train = unsafe_df.loc[train_mask, features].values
    X_test = unsafe_df.loc[test_mask, features].values
    
    # 3.1 极差标准化 (仅用训练集 fit)
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    # 对于 2月，只做 transform
    X_test_norm = scaler.transform(X_test) if len(X_test) > 0 else []
    
    # 3.2 熵权法 (EWM) - 仅用训练集计算客观权重
    X_train_eps = X_train_norm + 1e-5 
    P = X_train_eps / X_train_eps.sum(axis=0)
    e = - (1.0 / np.log(len(X_train))) * np.sum(P * np.log(P), axis=0)
    d = 1.0 - e
    w_ewm = d / d.sum()

    # 3.3 层次分析法 (AHP)
    ahp_matrix = np.array([
        [1,   2,   1, 1/2], 
        [1/2, 1, 1/2, 1/3], 
        [1,   2,   1, 1/2], 
        [2,   3,   2,   1]  
    ])
    eigvals, eigvecs = np.linalg.eig(ahp_matrix)
    w_ahp = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    w_ahp = w_ahp / np.sum(w_ahp)

    # 3.4 组合权重
    w_combined = (w_ewm * w_ahp) / np.sum(w_ewm * w_ahp)
    
    # 分别计算得分
    unsafe_df.loc[train_mask, 'Risk_Score'] = np.dot(X_train_norm, w_combined)
    if len(X_test) > 0:
        unsafe_df.loc[test_mask, 'Risk_Score'] = np.dot(X_test_norm, w_combined)


    # ==========================================
    # 4. ★ 核心分离：基于训练集的 K-Means 聚类 ★
    # ==========================================
    print("正在执行 K-Means 自适应边界划分 (仅基于1,3月数据)...")
    n_clusters = 3 if len(X_train) >= 3 else len(X_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # 仅用训练集 fit
    kmeans.fit(unsafe_df.loc[train_mask, ['Risk_Score']])
    
    # 分别对训练集和测试集 predict
    unsafe_df.loc[train_mask, 'Cluster'] = kmeans.predict(unsafe_df.loc[train_mask, ['Risk_Score']])
    if len(X_test) > 0:
        unsafe_df.loc[test_mask, 'Cluster'] = kmeans.predict(unsafe_df.loc[test_mask, ['Risk_Score']])
    
    # 中心排序映射
    centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)
    mapping = {sorted_idx[0]: 1, sorted_idx[1]: 2, sorted_idx[2]: 3}
    unsafe_df['Risk_Level'] = unsafe_df['Cluster'].map(mapping)
    
    label_map = {1: '低风险 (Low)', 2: '中风险 (Medium)', 3: '高风险 (High)'}
    unsafe_df['Status'] = unsafe_df['Risk_Level'].map(label_map)

    # 合并回总表
    df_daily.set_index('Date', inplace=True)
    unsafe_df.set_index('Date', inplace=True)
    df_daily.update(unsafe_df)
    df_daily.reset_index(inplace=True)


# ==========================================
# 5. 疯狂输出：7张融合样本外验证视角的学术大图
# ==========================================
print("\n正在火力全开渲染学术图表...")
color_dict = {'安全 (Safe)': '#2ca02c', '低风险 (Low)': '#f1c40f', '中风险 (Medium)': '#e67e22', '高风险 (High)': '#d62728'}

# 【图1】AHP专家判断矩阵热力图
plt.figure(figsize=(6, 5))
sns.heatmap(ahp_matrix, annot=True, fmt=".2f", cmap="YlOrRd", 
            xticklabels=['M_max', 'M_mean', 'D_total', 'D_cont'], 
            yticklabels=['M_max', 'M_mean', 'D_total', 'D_cont'], 
            cbar_kws={'label': '相对重要性'}, annot_kws={"size": 14, "weight": "bold"})
plt.title("AHP 主观判断矩阵 (一致性 CR<0.1)", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

# 【图2】EWM 熵值与信息效用双轴图 (标注仅基于训练集)
fig2, ax2_1 = plt.subplots(figsize=(8, 5))
ax2_2 = ax2_1.twinx()
x_pos = np.arange(len(feature_names))
ax2_1.bar(x_pos, e, color='#95a5a6', alpha=0.6, width=0.4, edgecolor='black', label='信息熵 (e)')
ax2_2.plot(x_pos, d, color='#e74c3c', marker='D', markersize=10, linewidth=3, label='信息效用度 (d)')
ax2_1.set_ylabel('信息熵 (e)', fontsize=12)
ax2_2.set_ylabel('信息效用度 (d)', fontsize=12, color='#e74c3c')
ax2_1.set_xticks(x_pos)
ax2_1.set_xticklabels(['M_max', 'M_mean', 'D_total', 'D_cont'], fontsize=12, fontweight='bold')
plt.title("EWM 熵权法特征离散度测算 (基于 1/3月 训练集)", fontsize=14, fontweight='bold', pad=15)
lines, labels = ax2_1.get_legend_handles_labels()
lines2, labels2 = ax2_2.get_legend_handles_labels()
ax2_2.legend(lines + lines2, labels + labels2, loc='center left')
plt.tight_layout()

# 【图3】主客观权重组合对比图
fig3, ax3 = plt.subplots(figsize=(9, 5))
width = 0.25
ax3.bar(x_pos - width, w_ahp, width, label='AHP 主观权重', color='#3498db', edgecolor='black')
ax3.bar(x_pos, w_ewm, width, label='EWM 客观权重(训练集)', color='#9b59b6', edgecolor='black')
ax3.bar(x_pos + width, w_combined, width, label='组合权重', color='#e74c3c', edgecolor='black')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['M_max', 'M_mean', 'D_total', 'D_cont'], fontsize=12, fontweight='bold')
ax3.set_ylabel('权重系数', fontsize=12)
ax3.set_title("AHP-EWM 组合赋权机制结果对比", fontsize=14, fontweight='bold', pad=15)
ax3.legend()
plt.tight_layout()

# 【图4】1, 2, 3月各等级天数占比对比 (三饼图，展示泛化能力)
fig4, axes_pie = plt.subplots(1, 3, figsize=(18, 6))
months_name = [1, 2, 3]
for i, m in enumerate(months_name):
    df_m = df_daily[df_daily['Month'] == m]['Status'].value_counts()
    axes_pie[i].pie(df_m.values, labels=df_m.index, autopct='%1.1f%%', startangle=140, 
                    colors=[color_dict.get(x, '#333') for x in df_m.index], wedgeprops={'edgecolor': 'black'})
    title_suffix = "(训练集)" if m in [1, 3] else "(独立测试集)"
    axes_pie[i].set_title(f"{m}月份 风险占比 {title_suffix}", fontsize=14, fontweight='bold')
fig4.suptitle("训练集与测试集 (1-3月) 水质评价泛化表现对比", fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

# 【图5】K-Means 聚类风险得分时序散点图 (区分 Train 和 Test 形状)
fig5, ax5 = plt.subplots(figsize=(14, 6))
df_plot = df_daily[df_daily['Status'] != '安全 (Safe)'].copy()
df_plot['Date'] = pd.to_datetime(df_plot['Date'])

# 画训练集 (圆圈)
for status in ['低风险 (Low)', '中风险 (Medium)', '高风险 (High)']:
    subset_train = df_plot[(df_plot['Status'] == status) & (df_plot['Month'].isin([1,3]))]
    if len(subset_train) > 0:
        ax5.scatter(subset_train['Date'], subset_train['Risk_Score'], color=color_dict[status], 
                    marker='o', s=120, edgecolor='k', label=f'{status} (Train)')
# 画测试集 (五角星)
for status in ['低风险 (Low)', '中风险 (Medium)', '高风险 (High)']:
    subset_test = df_plot[(df_plot['Status'] == status) & (df_plot['Month'] == 2)]
    if len(subset_test) > 0:
        ax5.scatter(subset_test['Date'], subset_test['Risk_Score'], color=color_dict[status], 
                    marker='*', s=250, edgecolor='k', label=f'{status} (Test 2月)')

for center in centers:
    ax5.axhline(y=center, color='gray', linestyle='--', alpha=0.5)
ax5.set_title("综合风险得分 (Risk Score) 聚类边界与泛化前向预测展示", fontsize=16, fontweight='bold', pad=15)
ax5.set_ylabel('AHP-EWM 综合得分', fontsize=13)
ax5.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 【图6】特征雷达图 (仅展示训练出的聚类中心的特征画像)
if len(X_train) > 0:
    radar_data = pd.DataFrame(X_train_norm, columns=features)
    radar_data['Risk_Level'] = unsafe_df[unsafe_df['Month'].isin([1,3])]['Risk_Level'].values
    radar_avg = radar_data.groupby('Risk_Level').mean()
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    
    fig6, ax6 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for level, color in zip([1, 2, 3], ['#f1c40f', '#e67e22', '#d62728']):
        if level in radar_avg.index:
            values = radar_avg.loc[level].tolist()
            values += values[:1]
            ax6.plot(angles, values, color=color, linewidth=2, label=label_map[level])
            ax6.fill(angles, values, color=color, alpha=0.25)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(['M_max', 'M_mean', 'D_total', 'D_cont'], fontsize=12, fontweight='bold')
    ax6.set_title("各风险级别特征画像 (基于训练集拟合边界)", fontsize=15, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

plt.show()

# ==========================================
# 6. 导出包含全季度 (1-3月) 标注了 Train/Test 的评估报表
# ==========================================
print("\n正在生成第一季度全域风险评估报表并导出 Excel...")

df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.strftime('%Y-%m-%d')
output_report = df_daily[['Date', 'Dataset', 'M_max', 'D_total', 'D_cont', 'Risk_Score', 'Status']].copy()

output_report.rename(columns={
    'Date': '日期',
    'Dataset': '数据集阵营',
    'M_max': '日最大超标幅度(NTU)',
    'D_total': '日总超标时长(H)',
    'D_cont': '最大连续超标(H)',
    'Risk_Score': '综合风险得分',
    'Status': '最终评定等级'
}, inplace=True)

output_report['日最大超标幅度(NTU)'] = output_report['日最大超标幅度(NTU)'].round(3)
output_report['综合风险得分'] = output_report['综合风险得分'].fillna(0).round(4)

excel_filename = 'Question4_Q1_Evaluation_With_TestSet.xlsx'
output_report.to_excel(excel_filename, index=False)

print(f"\n✅ 考核大通关！包含2月独立测试集的评估报表已导出至: {excel_filename}")