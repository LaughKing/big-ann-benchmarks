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
MARKER_SIZE = 8 # 调整标记大小
MARKER_EVERY_CDF = 10 # 在CDF图上标记的频率

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
# 修改输出目录以反映新内容
output_dir = "latency_cdf_plots_custom_style"
os.makedirs(output_dir, exist_ok=True)

# 2. 加载数据
try:
    data = pd.read_csv(input_file)
    data.columns = data.columns.str.strip()
    print("CSV文件加载成功！")
except FileNotFoundError:
    print(f"错误：找不到文件 '{input_file}'。")
    exit()

# 3. 识别 batchLatency 列
# 修改正则表达式以匹配 'batchLatency_'
latency_cols = sorted([col for col in data.columns if re.match(r'batchLatency_\d+', col)], 
                         key=lambda x: int(x.split('_')[1]))

# --- 更新的诊断信息 ---
if not latency_cols:
    print("错误：在文件中没有找到 'batchLatency_N' 格式的列。")
    exit()
else:
    # 打印找到的列数
    print(f"诊断信息：在文件中找到了 {len(latency_cols)} 个 'batchLatency' 数据点用于绘制CDF。")

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
        
        # --- CDF 计算逻辑 ---
        # 1. 获取所有延迟数据点
        latency_values = row[latency_cols].values.astype(float)
        
        # --- 数据清洗修正：移除NaN值 ---
        # NaN值通常由CSV中的空单元格产生，会导致CDF计算不完整
        latency_values = latency_values[~np.isnan(latency_values)]
        
        # 如果过滤后没有数据，则跳过该算法
        if len(latency_values) == 0:
            continue
            
        # 2. 对延迟数据进行排序，作为X轴
        x_cdf = np.sort(latency_values)
        
        # 3. 创建对应的Y轴 (0到1的累积概率)
        y_cdf = np.arange(1, len(x_cdf) + 1) / len(x_cdf)
        # --- CDF 计算结束 ---

        style = algo_to_style.get(algorithm_name, {'color': 'black', 'marker': 'x'}) # 获取样式
        
        # 绘制CDF线条
        plt.plot(x_cdf, y_cdf, color=style['color'], marker=style['marker'], 
                 markevery=MARKER_EVERY_CDF, markersize=MARKER_SIZE, 
                 linewidth=LINE_WIDTH, label=algorithm_name)

    # --- 应用专业的图表格式 ---
    # 更新Y轴标签和标题
    plt.xlabel("Batch Latency (ms)", fontproperties=LABEL_FP)
    plt.ylabel("CDF", fontproperties=LABEL_FP)
    plt.title(f"Latency CDF on '{dataset_name}'", fontproperties=LABEL_FP, weight=LABEL_WEIGHT, pad=20)
    
    # 将X轴设置为对数刻度，这对于观察延迟分布通常更有效
    plt.xscale('log')
    
    # --- Y轴范围修正 ---
    # 强制将Y轴的范围设置为0到1.0，以确保显示完整
    plt.ylim(0, 1.05)
    
    plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)
    plt.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.4)
    
    # 移除图例
    # plt.legend(prop=LEGEND_FP, loc='best')
    
    # 使用标准的 tight_layout
    plt.tight_layout() 
    
    # 5. 保存图表
    # 更新保存的文件名
    safe_dataset_name = "".join([c for c in dataset_name if c.isalnum() or c in ('_','-')]).rstrip()
    save_path = os.path.join(output_dir, f"latency_cdf_{safe_dataset_name}.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close()

print("\n所有图表已生成完毕！")
print(f"文件保存在目录: '{output_dir}'")