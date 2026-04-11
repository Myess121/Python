# config.py
# -*- coding: utf-8 -*-
"""
多无人机数量调参配置文件（优化版）
"""

# ======================== 角色配置 ========================
# 可选: master, worker1, worker2, stabler, multi_k
ROLE = "multi_k"

# ======================== 多K值调参配置 ========================
# 要测试的无人机数量列表
K_LIST = [3, 4]

# 每个K值的试验次数（建议：快速测试用10-20，正式运行用30-50）
N_TRIALS_PER_K = 50

# 每次评估的仿真次数
N_SIM_PER_EVAL = 50

# 随机种子
RANDOM_SEED = 42

# ======================== 各K值的搜索空间 ========================
# 基础搜索空间（适用于所有K值）
BASE_SEARCH_SPACE = {
    'H': {'type': 'int', 'low': 3, 'high': 6, 'step': 1},
    'pop_size': {'type': 'int', 'low': 15, 'high': 35, 'step': 2},
    'max_iter': {'type': 'int', 'low': 20, 'high': 40, 'step': 2},
    'w_start': {'type': 'float', 'low': 0.70, 'high': 0.95},
    'w_end': {'type': 'float', 'low': 0.10, 'high': 0.40},
    'c1': {'type': 'float', 'low': 1.0, 'high': 2.2},
    'c2': {'type': 'float', 'low': 0.8, 'high': 2.0},
    'penalty_coef': {'type': 'float', 'low': 100, 'high': 2000},  # 优化范围
}

# 针对不同K值调整的搜索空间
SEARCH_SPACE_BY_K = {
    2: {
        'H': {'type': 'int', 'low': 5, 'high': 8, 'step': 1},
        'pop_size': {'type': 'int', 'low': 20, 'high': 40, 'step': 2},
        'max_iter': {'type': 'int', 'low': 25, 'high': 45, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 100, 'high': 1500},
    },
    3: {
        'H': {'type': 'int', 'low': 4, 'high': 7, 'step': 1},
        'pop_size': {'type': 'int', 'low': 18, 'high': 38, 'step': 2},
        'max_iter': {'type': 'int', 'low': 22, 'high': 42, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 100, 'high': 1500},
    },
    4: {
        'H': {'type': 'int', 'low': 4, 'high': 6, 'step': 1},
        'pop_size': {'type': 'int', 'low': 15, 'high': 35, 'step': 2},
        'max_iter': {'type': 'int', 'low': 20, 'high': 40, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 150, 'high': 1800},
    },
    5: {
        'H': {'type': 'int', 'low': 3, 'high': 6, 'step': 1},
        'pop_size': {'type': 'int', 'low': 15, 'high': 35, 'step': 2},
        'max_iter': {'type': 'int', 'low': 20, 'high': 40, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 200, 'high': 2000},
    },
    6: {
        'H': {'type': 'int', 'low': 3, 'high': 5, 'step': 1},
        'pop_size': {'type': 'int', 'low': 15, 'high': 30, 'step': 2},
        'max_iter': {'type': 'int', 'low': 20, 'high': 35, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 200, 'high': 2000},
    },
    7: {
        'H': {'type': 'int', 'low': 3, 'high': 5, 'step': 1},
        'pop_size': {'type': 'int', 'low': 20, 'high': 35, 'step': 2},
        'max_iter': {'type': 'int', 'low': 25, 'high': 40, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 300, 'high': 2000},
    },
    8: {
        'H': {'type': 'int', 'low': 2, 'high': 5, 'step': 1},
        'pop_size': {'type': 'int', 'low': 20, 'high': 40, 'step': 2},
        'max_iter': {'type': 'int', 'low': 25, 'high': 45, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 300, 'high': 2000},
    },
    9: {
        'H': {'type': 'int', 'low': 2, 'high': 4, 'step': 1},
        'pop_size': {'type': 'int', 'low': 25, 'high': 45, 'step': 2},
        'max_iter': {'type': 'int', 'low': 30, 'high': 50, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 400, 'high': 2000},
    },
}

# ======================== Optuna配置 ========================
OPTUNA_CONFIG = {
    'sampler': 'TPE',
    'pruner': 'MedianPruner',
    'n_startup_trials': 5,
    'n_ei_candidates': 24,
    'timeout': None,
}

# ======================== 仿真参数 ========================
SIMULATION_PARAMS = {
    'M': 16,
    'N': 23,
    'dx': 20.0,
    'dy': 20.0,
    'V_uav': 180.0,
    'max_flight_time': 40.0,
    'start_step': 23,
    'Pd': 0.98,
    'Pf': 0.02,
    'p_stay': 0.95,
    'p_move': 0.05,
}

# ======================== 默认PSO参数 ========================
DEFAULT_PSO_PARAMS = {
    'H': 4,
    'pop_size': 25,
    'max_iter': 30,
    'w_start': 0.85,
    'w_end': 0.25,
    'c1': 1.5,
    'c2': 1.5,
    'penalty_coef': 500,
    'use_pso': True
}

# ======================== 可视化配置 ========================
VISUALIZATION_CONFIG = {
    'dpi': 150,
    'figsize_large': (16, 12),
    'figsize_medium': (14, 10),
    'figsize_small': (10, 8),
    'color_scheme': {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#18A999',
        'warning': '#F18F01',
        'danger': '#C73E1D',
    }
}


def get_search_space(K):
    """根据无人机数量获取完整的搜索空间"""
    search_space = BASE_SEARCH_SPACE.copy()

    if K in SEARCH_SPACE_BY_K:
        for param, config in SEARCH_SPACE_BY_K[K].items():
            if param in search_space:
                search_space[param].update(config)
            else:
                search_space[param] = config

    return search_space


def print_config():
    """打印当前配置"""
    print("\n" + "=" * 60)
    print("当前配置:")
    print("=" * 60)
    print(f"角色: {ROLE}")
    print(f"测试的K值列表: {K_LIST}")
    print(f"每个K值的试验次数: {N_TRIALS_PER_K}")
    print(f"每次评估仿真次数: {N_SIM_PER_EVAL}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)

    print("\n搜索空间配置:")
    for k in K_LIST:
        print(f"\nK = {k}:")
        space = get_search_space(k)
        for param, config in space.items():
            print(f"    {param}: {config['low']} ~ {config['high']}")


if __name__ == "__main__":
    print_config()