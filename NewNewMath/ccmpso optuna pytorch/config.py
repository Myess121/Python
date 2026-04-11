# config.py
# -*- coding: utf-8 -*-
"""
CC-MPSO协同搜索配置文件
"""

# ======================== 仿真参数 ========================
M, N = 16, 23  # 网格数
dx, dy = 20.0, 20.0  # 网格边长 (km)
V_uav = 180.0  # 无人机速度 (km/h)
dt = dx / V_uav  # 时间步长 (h)
max_flight_time = 40.0  # 最大续航 (h)
max_steps = int(max_flight_time / dt)  # 约360步
start_step = 23  # 第23步开始探测（前22步入场）

# 探测参数
Pd = 0.9  # 探测概率
Pf = 0.05  # 虚警概率

# 目标运动参数
p_stay = 0.95  # 留在原网格概率
p_move = 0.05  # 移动到邻域总概率

# ======================== CC-MPSO默认参数 ========================
DEFAULT_CCMPSO_PARAMS = {
    'H': 5,  # 规划步数
    'pop_size': 20,  # 粒子数
    'max_iter': 22,  # 迭代次数
    'coop_gap': 5,  # 协同间隔
    'w_start': 0.9075,  # 惯性权重起始
    'w_end': 0.2701,  # 惯性权重结束
    'c1': 1.8594,  # 认知系数
    'c2': 1.1079,  # 社会系数
    'penalty_coef': 1974.16,  # 碰撞惩罚系数
}

# ======================== 调参配置 ========================
# 要测试的无人机数量列表
K_LIST = [1,2,3,4]

# 每个K值的试验次数（Optuna trials）
N_TRIALS_PER_K = 50

# 每次评估的仿真次数
N_SIM_PER_EVAL = 50

# 随机种子
RANDOM_SEED = 42

# ======================== CC-MPSO搜索空间 ========================
# 基础搜索空间
BASE_SEARCH_SPACE = {
    'H': {'type': 'int', 'low': 3, 'high': 7, 'step': 1},
    'pop_size': {'type': 'int', 'low': 15, 'high': 35, 'step': 2},
    'max_iter': {'type': 'int', 'low': 15, 'high': 35, 'step': 2},
    'w_start': {'type': 'float', 'low': 0.70, 'high': 0.95},
    'w_end': {'type': 'float', 'low': 0.10, 'high': 0.45},
    'c1': {'type': 'float', 'low': 1.0, 'high': 2.2},
    'c2': {'type': 'float', 'low': 0.8, 'high': 2.0},
    'penalty_coef': {'type': 'float', 'low': 500, 'high': 3000},
}

# 针对不同K值调整的搜索空间
SEARCH_SPACE_BY_K = {
    5: {
        'H': {'type': 'int', 'low': 3, 'high': 6, 'step': 1},
        'pop_size': {'type': 'int', 'low': 15, 'high': 30, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 500, 'high': 2500},
    },
    6: {
        'H': {'type': 'int', 'low': 3, 'high': 6, 'step': 1},
        'pop_size': {'type': 'int', 'low': 15, 'high': 30, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 500, 'high': 2500},
    },
    7: {
        'H': {'type': 'int', 'low': 4, 'high': 7, 'step': 1},
        'pop_size': {'type': 'int', 'low': 18, 'high': 35, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 800, 'high': 3000},
    },
    8: {
        'H': {'type': 'int', 'low': 4, 'high': 7, 'step': 1},
        'pop_size': {'type': 'int', 'low': 20, 'high': 38, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 1000, 'high': 3500},
    },
    9: {
        'H': {'type': 'int', 'low': 5, 'high': 8, 'step': 1},
        'pop_size': {'type': 'int', 'low': 22, 'high': 40, 'step': 2},
        'penalty_coef': {'type': 'float', 'low': 1200, 'high': 4000},
    },
}

# ======================== Optuna配置 ========================
OPTUNA_CONFIG = {
    'n_startup_trials': 5,
    'n_ei_candidates': 24,
    'timeout': None,
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
    print("CC-MPSO协同搜索配置")
    print("=" * 60)
    print(f"网格尺寸: {M} × {N}")
    print(f"时间步长: {dt:.4f} 小时 ({dt * 60:.1f} 分钟)")
    print(f"最大步数: {max_steps}")
    print(f"入场步数: {start_step}")
    print(f"探测概率 Pd: {Pd}")
    print(f"虚警概率 Pf: {Pf}")
    print(f"目标停留概率: {p_stay}")
    print(f"测试K值: {K_LIST}")
    print(f"每K试验次数: {N_TRIALS_PER_K}")
    print(f"每评估仿真次数: {N_SIM_PER_EVAL}")
    print("=" * 60)

    print("\nCC-MPSO默认参数:")
    for k, v in DEFAULT_CCMPSO_PARAMS.items():
        print(f"  {k} = {v}")

    print("\n搜索空间:")
    for k in K_LIST:
        print(f"  K={k}: {list(get_search_space(k).keys())}")


if __name__ == "__main__":
    print_config()