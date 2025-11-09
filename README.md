# 项目结构详解
    project/  
    configs/         # 配置（yaml/json）   
    data/            # 数据或数据链接（README 说明获取方式）
    src/             # 训练/评测/部署核心代码 
    scripts/         # 训练/评测/导出/部署一键脚本 
    notebooks/       # 探索性分析
    logs             # 日志
    results          #可视化结果
    eval/            # 评测集与评测脚本 
    README.md        # 复现指南（三步走）


### week1复现指南
+ src\main1.py是主程序（暂时未配备输出日志和曲线） 
+ results\1wine_corr.png为绘制wine特征间相关性热力图
+ results\1wine_mean_std.png为绘制wine特征均值和方差柱状图
