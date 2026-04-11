# -*- coding: utf-8 -*-
"""
配置文件 - 稳定版
"""

# ======================== 角色配置 ========================
ROLE = "worker1"  # 可选: master, worker1, worker2

# ======================== 搜索空间 ========================
if ROLE == "master":
    SEARCH_SPACE = {
        'H': {'type': 'int', 'low': 4, 'high': 8, 'step': 1},
        'pop_size': {'type': 'int', 'low': 10, 'high': 20, 'step': 1},
        'max_iter': {'type': 'int', 'low': 15, 'high': 25, 'step': 1},
        'w_start': {'type': 'float', 'low': 0.7, 'high': 0.95},
        'w_end': {'type': 'float', 'low': 0.2, 'high': 0.5},
        'c1': {'type': 'float', 'low': 1.0, 'high': 2.0},
        'c2': {'type': 'float', 'low': 1.0, 'high': 2.0},
        'penalty_coef': {'type': 'float', 'low': 500, 'high': 3000},
    }
    N_TRIALS = 50  # 先跑50次试验
    N_SIM_PER_EVAL = 30  # 每次评估30次仿真（加快速度）

elif ROLE == "worker1":
    SEARCH_SPACE = {
        'H': {'type': 'int', 'low': 4, 'high': 8, 'step': 1},
        'pop_size': {'type': 'int', 'low': 10, 'high': 20, 'step': 1},
        'max_iter': {'type': 'int', 'low': 15, 'high': 25, 'step': 1},
        'w_start': {'type': 'float', 'low': 0.7, 'high': 0.95},
        'w_end': {'type': 'float', 'low': 0.2, 'high': 0.5},
        'c1': {'type': 'float', 'low': 1.0, 'high': 2.0},
        'c2': {'type': 'float', 'low': 1.0, 'high': 2.0},
        'penalty_coef': {'type': 'float', 'low': 500, 'high': 3000},
    }
    N_TRIALS = 40
    N_SIM_PER_EVAL = 25

elif ROLE == "worker2":
    SEARCH_SPACE = {
        'H': {'type': 'int', 'low': 4, 'high': 8, 'step': 1},
        'pop_size': {'type': 'int', 'low': 10, 'high': 20, 'step': 1},
        'max_iter': {'type': 'int', 'low': 15, 'high': 25, 'step': 1},
        'w_start': {'type': 'float', 'low': 0.7, 'high': 0.95},
        'w_end': {'type': 'float', 'low': 0.2, 'high': 0.5},
        'c1': {'type': 'float', 'low': 1.0, 'high': 2.0},
        'c2': {'type': 'float', 'low': 1.0, 'high': 2.0},
        'penalty_coef': {'type': 'float', 'low': 500, 'high': 3000},
    }
    N_TRIALS = 40
    N_SIM_PER_EVAL = 25

# ======================== 固定参数 ========================
FIXED_PARAMS = {
    'K': 7,
}

# ======================== Optuna配置 ========================
OPTUNA_CONFIG = {
    'sampler': 'TPE',
    'pruner': 'MedianPruner',
    'timeout': None,
}