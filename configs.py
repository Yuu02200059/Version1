# configs.py

# 默认配置 (作为一个兜底，防止数据集没在列表里报错)
DEFAULT_CONFIG = {
    'feature_dim': 512,
    'semantic_dim': 128,
    'hidden_dim': 512,  # Encoder中间层大小
    'lambda_rec': 1.0,  # 重构损失权重
    'lambda_con': 0.1,  # 对比损失权重
    'lambda_global': 1.0,  # ITGG权重
    'lr': 1e-3,
    'batch_size': 256,
    'epochs': 100
}

CONFIGS = {
    # === 场景 A: 高维数据 (HandWritten, Caltech, etc.) ===
    'HandWritten': {
        'feature_dim': 128,  # 增大特征维度以容纳高维输入
        'semantic_dim': 128,
        'hidden_dim': 512,  # 增大中间层容量
        # 保持结构损失和弱引导
        'lambda_rec': 1,  # 高维数据需要弱重构，抑制噪声
        'lambda_struct': 1.0,  # 结构一致性必须强力驱动
        'lambda_global': 0.01,#强对齐（对比损失
        'lambda_div': 0.1,
        'lr': 2e-3,  # 适度降低学习率
        #handwritten缺失率0.1时只需要2e-3 0.3 2e-3,0.5 2e-3 而缺失率0.7时需要2e-3！
        'batch_size': 256,
        'epochs': 200
    },

    # === 场景 B: 低维特征数据 (Scene_15, LandUse, etc.) ===
    'Scene_15': {
        'feature_dim': 128,  # 输入维度小，特征维度必须小
        'semantic_dim': 64,
        'hidden_dim': 256,  # 中间层适度瘦身
        'lambda_rec': 1,
        'lambda_struct': 1,
        'lambda_global': 0.01,
        'lambda_div': 0.1,
        'lr': 1e-3,  # 小网络，学习率调低防止震荡
        'batch_size': 256,
        'epochs': 200,
    },
    'Caltech101_7': {
        'feature_dim': 256,  # 增大特征维度以容纳高维输入
        'semantic_dim': 128,
        'hidden_dim': 512,  # 增大中间层容量
        # 保持结构损失和弱引导
        'lambda_rec': 1,  # 高维数据需要弱重构，抑制噪声
        'lambda_struct': 1.0,  # 结构一致性必须强力驱动
        'lambda_global': 0.01,#强对齐（对比损失
        'lambda_div': 0.1,
        'lr': 5e-4,  # 适度降低学习率
        #0.1使用学习率4e-4，0.3使用5e-4 0.5使用4e-4,0.7使用5e-4
        'batch_size': 256,
        'epochs': 200
    },
    'Caltech101_20': {
        'feature_dim': 256,  # 增大特征维度以容纳高维输入
        'semantic_dim': 128,
        'hidden_dim': 512,  # 增大中间层容量
        # 保持结构损失和弱引导
        'lambda_rec': 1,  # 高维数据需要弱重构，抑制噪声
        'lambda_struct': 1.0,  # 结构一致性必须强力驱动
        'lambda_global': 0.01,  # 强对齐（对比损失
        'lambda_div': 0.1,
        'lr': 5e-4,  # 适度降低学习率
        # 0.1使用学习率4e-4，0.3使用5e-4 0.5使用4e-4,0.7使用5e-4
        'batch_size': 256,
        'epochs': 200
    },

    # 你可以继续添加其他数据集...
    'BDGP': {
        'feature_dim': 512,
        'semantic_dim': 64,
        'hidden_dim': 256,
        'lambda_rec': 1.0,
        'lr': 1e-3
    }
}
def get_config(dataset_name):
    """根据数据集名称返回配置字典，如果不存在则返回默认"""
    if dataset_name in CONFIGS:
        print(f"✅ 已加载 [{dataset_name}] 的专属配置")
        return CONFIGS[dataset_name]
    else:
        print(f"⚠️ 未找到 [{dataset_name}] 的专属配置，使用默认配置")
        return DEFAULT_CONFIG