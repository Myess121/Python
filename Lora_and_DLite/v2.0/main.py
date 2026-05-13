# main.py
"""
高保真无人机通信感知路径规划 - 蒙特卡洛仿真主程序 (顶刊大修版)
用于生成论文核心对比表格与图表数据。
"""
import time
import math
import random
import numpy as np

# 导入黄金 5 阵型
from planners.astar import AStar
from planners.astar_comm import AStarComm
from planners.ca_rrt_star import CARRTStar
from planners.dstar_comm import DStarComm
from environment import calculate_physical_sinr, get_per_from_sinr


# ================= 1. 仿真环境类定义 =================
class GridMap:
    def __init__(self, width, height, obstacle_ratio=0.15):
        self.width = width
        self.height = height
        self.obstacles = set()
        self._generate_obstacles(obstacle_ratio)

    def _generate_obstacles(self, ratio):
        """随机生成城市建筑障碍物"""
        num_obs = int(self.width * self.height * ratio)
        while len(self.obstacles) < num_obs:
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            # 避开起点(2,2)和终点(28,28)附近
            if not (x < 5 and y < 5) and not (x > self.width - 6 and y > self.height - 6):
                self.obstacles.add((x, y))

    def is_collision(self, pos):
        if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
            return True
        return pos in self.obstacles


class DynamicJammer:
    def __init__(self, x, y, power=20):
        self.pos = [x, y]
        self.old_pos = [x, y]  # 【新增】记录上一次的位置
        self.power = power

    def random_walk(self, grid_map):
        """干扰源随机游走模拟"""
        self.old_pos = list(self.pos)  # 【新增】在移动前，保存旧位置

        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)])
        new_x, new_y = self.pos[0] + dx, self.pos[1] + dy
        if 0 <= new_x < grid_map.width and 0 <= new_y < grid_map.height:
            if not grid_map.is_collision((new_x, new_y)):
                self.pos = [new_x, new_y]


# ================= 2. 评估函数 =================
def evaluate_path_per(path, gs_pos, jammers):
    """计算整条路径上的平均误包率 (Average PER)"""
    if not path:
        return 1.0  # 没找到路，视为 100% 丢包
    total_per = 0.0
    for pos in path:
        sinr = calculate_physical_sinr(pos, gs_pos, jammers)
        total_per += get_per_from_sinr(sinr)
    return total_per / len(path)


# ================= 3. 主仿真循环 =================
def run_simulation(num_runs=5):  # 论文里建议改成 30 次，这里设 5 次为了快速测试
    print(f"🚀 开始蒙特卡洛仿真 (共 {num_runs} 轮)...\n")

    # 存储结果的字典
    results = {
        "1. Pure A* (几何基线)": {"len": [], "per": [], "time": []},
        "2. Standard D* Lite": {"len": [], "per": [], "time": []},
        "3. A*-Comm (重算基线)": {"len": [], "per": [], "time": []},
        "4. CA-RRT* (采样基线)": {"len": [], "per": [], "time": []},
        "5. D* Lite-Comm (本文)": {"len": [], "per": [], "time": []}
    }

    for run in range(num_runs):
        print(f"--- 正在运行第 {run + 1}/{num_runs} 轮测试 ---")

        # 1. 随机生成 30x30 复杂地图和干扰源
        grid_map = GridMap(50, 50, obstacle_ratio=0.15)
        s_start, s_goal = (2, 2), (48, 48)
        gs_pos = (25, 25)

        # 【修改这里】降低干扰源的功率，给无人机绕路的可能
        jammers = [
            DynamicJammer(random.randint(15, 35), random.randint(15, 35), power=15),
            DynamicJammer(random.randint(15, 35), random.randint(15, 35), power=15)
        ]

        # 2. 初始化算法
        std_dstar = DStarComm(s_start, s_goal)
        std_dstar.alpha = 0.0  # 设为瞎子
        prop_dstar = DStarComm(s_start, s_goal)
        prop_dstar.alpha = 10.0  # 开启通信感知

        planners = {
            "1. Pure A* (几何基线)": AStar(s_start, s_goal),
            "2. Standard D* Lite": std_dstar,
            "3. A*-Comm (重算基线)": AStarComm(s_start, s_goal),
            "4. CA-RRT* (采样基线)": CARRTStar(s_start, s_goal),
            "5. D* Lite-Comm (本文)": prop_dstar
        }

        # 3. 首次规划
        for name, planner in planners.items():
            planner.plan(grid_map, gs_pos, jammers)

        # 4. 触发动态环境：干扰源移动
        for j in jammers:
            j.random_walk(grid_map)
            j.random_walk(grid_map)  # 移动两步

            # ====== 替换原来的 dynamic_replan 调用逻辑 ======
            # 5. 执行动态重规划并记录核心指标
            for name, planner in planners.items():
                t0 = time.time()
                if hasattr(planner, 'dynamic_replan'):
                    # 关键修复：如果是 D* Lite，严格限制局部更新半径为 5 格！
                    if isinstance(planner, DStarComm):
                        path = planner.dynamic_replan(impact_radius=5)
                    else:
                        path = planner.dynamic_replan()
                else:
                    path = planner.plan(grid_map, gs_pos, jammers)
                t1 = time.time()

            replan_time_ms = (t1 - t0) * 1000
            path_len = len(path) if path else 0
            avg_per = evaluate_path_per(path, gs_pos, jammers)

            # 记录数据
            results[name]["time"].append(replan_time_ms)
            results[name]["len"].append(path_len)
            results[name]["per"].append(avg_per)

    # ================= 4. 打印论文对比表格数据 =================
    print("\n" + "=" * 70)
    print("📊 仿真结果汇总 (可直接填入论文 Table 1)")
    print("=" * 70)
    print(f"{'算法名称':<25} | {'平均重规划耗时 (ms)':<20} | {'平均误包率 (PER)':<15} | {'平均路径长度'}")
    print("-" * 70)

    for name, data in results.items():
        avg_time = np.mean(data['time'])
        std_time = np.std(data['time'])

        avg_per = np.mean(data['per']) * 100  # 转为百分比
        std_per = np.std(data['per']) * 100

        avg_len = np.mean(data['len'])

        print(
            f"{name:<22} | {avg_time:>6.1f} ± {std_time:>4.1f} ms       | {avg_per:>6.2f}% ± {std_per:>4.2f}% | {avg_len:>6.1f}")
    print("=" * 70)


if __name__ == "__main__":
    run_simulation(num_runs=5)  # 试跑 5 次，确保不报错