import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def init_logger(log_path):
    """初始化日志"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def z_score_python(data):
    """Python原生循环实现z-score标准化（用于对比）"""
    n_samples, n_features = data.shape
    result = data.copy()  # 直接复制整个数组
    
    # 对每个特征进行标准化
    for i in range(n_features):
        # 一次性计算均值
        feature_sum = 0.0
        for j in range(n_samples):
            feature_sum += result[j, i]
        mean = feature_sum / n_samples
        
        # 计算方差
        squared_sum = 0.0
        for j in range(n_samples):
            result[j, i] -= mean  # 同时减去均值
            squared_sum += result[j, i] * result[j, i]
        
        # 计算标准差并规范化
        std = (squared_sum / n_samples) ** 0.5 + 1e-8
        for j in range(n_samples):
            result[j, i] /= std
            
    return result

def z_score_numpy(data):
    """Numpy向量化实现z-score标准化"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def plot_mean_std(data, feature_names, save_path):
    """绘制各特征的均值/方差图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    x = np.arange(len(feature_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mean, width, label='均值')
    ax.bar(x + width/2, std, width, label='方差')
    
    ax.set_xlabel('特征')
    ax.set_ylabel('值')
    ax.set_title('各特征的均值与方差')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_corr_heatmap(data, feature_names, save_path):
    """绘制特征相关热力图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    corr = np.corrcoef(data, rowvar=False)  # 计算特征间相关性
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', 
                xticklabels=feature_names, 
                yticklabels=feature_names)
    plt.title('特征相关热力图')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()