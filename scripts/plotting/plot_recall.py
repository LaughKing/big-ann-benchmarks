import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import matplotlib
import re

# --- 沿用您喜欢的专业绘图设置 ---
# 不再强制指定特定字体，以避免 "font not found" 警告
TICK_FONT_SIZE = 18
LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 12
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

LABEL_WEIGHT = 'bold'
LINE_WIDTH = 2.0
MARKER_SIZE = 9 # 调整标记大小

# 应用字体和PDF导出设置
try:
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
    matplotlib.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
    # 不再设置 font.family，让matplotlib使用默认字体
    matplotlib.rcParams['pdf.fonttype'] = 42
except Exception as e:
    print(f"字体设置警告: {e}")
    print("将使用系统默认字体。")

# --- 核心修正：根据您的图片定义固定的颜色和符号 ---
# 1. 定义算法的分类
categories = {
    'Tree': ['SPTAG'],
    'LSH': ['LSH', 'LSHAPG'],
    'Graph': ['NSW', 'HNSW', 'MNRU', 'FRESHDISK', 'CUFE', 'PYANNS'],
    'Clustering': ['PQ', 'ONLINEPQ', 'IVFPQ', 'PUCK', 'SCANN']
}

# 2. 定义每个算法对应的样式
algo_to_style = {
    # Tree
    'SPTAG': {'color': 'red', 'marker': 'D'},
    # LSH
    'LSH': {'color': 'blue', 'marker': 'o'},
    'LSHAPG': {'color': 'darkblue', 'marker': 'o'},
    # Graph
    'NSW': {'color': 'saddlebrown', 'marker': '>'},
    'HNSW': {'color': 'black', 'marker': '*'},
    'MNRU': {'color': 'green', 'marker': 's'},
    'FRESHDISK': {'color': 'purple', 'marker': '^'},
    'CUFE': {'color': 'cyan', 'marker': 'v'},
    'PYANNS': {'color': 'teal', 'marker': 'D'},
    # Clustering
    'PQ': {'color': 'gray', 'marker': 'p'},
    'ONLINEPQ': {'color': 'darkviolet', 'marker': '<'},
    'IVFPQ': {'color': 'orange', 'marker': 'X'},
    'PUCK': {'color': 'lime', 'marker': 'o'},
    'SCANN': {'color': 'darkkhaki', 'marker': '+'},
}
# --- 修正结束 ---

# --- 主要绘图逻辑 ---

# 1. 定义文件路径和输出目录
input_file = "processed_data.csv"
output_dir = "recall_trajectory_plots_custom_style"
os.makedirs(output_dir, exist_ok=True)

# 2. 加载数据
try:
    data = pd.read_csv(input_file)
    data.columns = data.columns.str.strip()
    print("CSV文件加载成功！")
except FileNotFoundError:
    print(f"错误：找不到文件 '{input_file}'。")
    exit()

# 3. 识别 batchRecall 列
recall_cols = sorted([col for col in data.columns if re.match(r'batchRecall_\d+', col)], 
                     key=lambda x: int(x.split('_')[1]))

# --- 新增的诊断信息 ---
if not recall_cols:
    print("错误：在文件中没有找到 'batchRecall_N' 格式的列。")
    exit()
else:
    # 打印找到的列数，帮助您确认数据是否完整
    print(f"诊断信息：在文件中找到了 {len(recall_cols)} 个 'batchRecall' 数据点 (X轴范围将是 0 到 {len(recall_cols)-1}).")

# 4. 为每个数据集循环绘图
unique_datasets = data['dataset'].unique()
print(f"检测到 {len(unique_datasets)} 个数据集，开始生成图表...")

for dataset_name in unique_datasets:
    print(f"正在处理数据集: {dataset_name}...")
    
    # 调整图表尺寸
    plt.figure(figsize=(12, 8)) 
    
    subset_df = data[data['dataset'] == dataset_name]
    
    for index, row in subset_df.iterrows():
        algorithm_name = row['algorithm']
        
        y_values = row[recall_cols].values.astype(float)
        x_values = np.arange(len(y_values))
        
        style = algo_to_style.get(algorithm_name, {'color': 'black', 'marker': 'x'}) # 获取样式
        
        # 绘制线条
        plt.plot(x_values, y_values, color=style['color'], marker=style['marker'], 
                 markevery=10, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=algorithm_name)

    # --- 应用专业的图表格式 ---
    plt.xlabel("Query Sequence Index", fontproperties=LABEL_FP)
    plt.ylabel("Batches", fontproperties=LABEL_FP)
    plt.title(f"Algorithm Recall Trajectory on '{dataset_name}'", fontproperties=LABEL_FP, weight=LABEL_WEIGHT, pad=20)
    plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)
    
    # --- 图例修改：移除图例 ---
    # plt.legend(prop=LEGEND_FP, loc='best') # 此行已注释，不再显示图例
    
    # 使用标准的 tight_layout
    plt.tight_layout() 
    
    # 5. 保存图表
    safe_dataset_name = "".join([c for c in dataset_name if c.isalnum() or c in ('_','-')]).rstrip()
    save_path = os.path.join(output_dir, f"recall_trajectory_{safe_dataset_name}.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close()

print("\n所有图表已生成完毕！")
print(f"文件保存在目录: '{output_dir}'")
