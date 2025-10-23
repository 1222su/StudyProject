import yaml
import numpy as np
import timeit
from sklearn.datasets import load_iris
import os
from utils import (
    init_logger, z_score_python, z_score_numpy,
    plot_mean_std, plot_corr_heatmap
)
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    # 1. 加载配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "configs", "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # 更新配置中的相对路径为绝对路径
    config['visualization']['save_dir'] = os.path.join(project_root, "results")
    config['log']['file'] = os.path.join(project_root, "logs", "experiment.log")
    
    # 2. 初始化日志
    logger = init_logger(config["log"]["file"])
    logger.info("===== 实验开始 =====")
    logger.info(f"加载配置: {config}")
    
    # 3. 加载数据（从sklearn获取鸢尾花数据集）
    iris = load_iris()
    data = iris.data  # 特征数据（150x4）
    feature_names = config["data"]["feature_cols"]
    logger.info(f"加载数据集: {config['data']['name']}，样本数: {data.shape[0]}，特征数: {data.shape[1]}")
    
    # 4. z-score标准化（对比Python循环与Numpy向量化速度）
    if config["preprocess"]["z_score"]:
        # 预热并清理缓存
        _ = z_score_python(data.copy())
        _ = z_score_numpy(data.copy())
        
        # 计时：Python循环
        python_time = min(
            timeit.timeit(
                lambda: z_score_python(data.copy()),
                number=1
            )
            for _ in range(100)  # 取100次中的最好成绩
        )
        
        # 计时：Numpy向量化
        numpy_time = min(
            timeit.timeit(
                lambda: z_score_numpy(data.copy()),
                number=1
            )
            for _ in range(100)  # 取100次中的最好成绩
        )
        
        # 计算加速比
        speedup = python_time / numpy_time
        logger.info(f"z-score标准化速度对比:")
        logger.info(f"Python循环: {python_time:.6f}秒/次")
        logger.info(f"Numpy向量化: {numpy_time:.6f}秒/次")
        logger.info(f"加速比: {speedup:.2f}x (≥10x要求: {'满足' if speedup>=10 else '不满足'})")
        
        # 验证结果一致性（确保两种方法结果相同）
        python_result = z_score_python(data)
        numpy_result = z_score_numpy(data)
        assert np.allclose(python_result, numpy_result, atol=1e-6), "两种方法结果不一致！"
        logger.info("z-score标准化结果验证通过")
    
    # 5. 可视化：均值/方差 + 相关热力图
    if config["visualization"]["mean_std"]:
        mean_std_path = os.path.join(config["visualization"]["save_dir"], "mean_std.png")
        plot_mean_std(data, feature_names, mean_std_path)
        logger.info(f"均值/方差图已保存至: {mean_std_path}")
    
    if config["visualization"]["corr_heatmap"]:
        corr_path = os.path.join(config["visualization"]["save_dir"], "corr_heatmap.png")
        plot_corr_heatmap(data, feature_names, corr_path)
        logger.info(f"相关热力图已保存至: {corr_path}")
    
    logger.info("===== 实验结束 =====")

if __name__ == "__main__":
    main() # 这里的代码只有在模块作为主程序运行时才会执行