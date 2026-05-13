# test_golden_five.py
import time

# 导入所有黄金阵型规划器
from planners.astar import AStar
from planners.astar_comm import AStarComm
from planners.ca_rrt_star import CARRTStar
from planners.dstar_comm import DStarComm


# ================= 1. 伪造极简测试环境 =================
class DummyGridMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def is_collision(self, pos):
        # 在地图中间造一堵小墙
        if pos[0] == 8 and 3 <= pos[1] <= 12:
            return True
        return False


class DummyJammer:
    def __init__(self, x, y):
        self.pos = [x, y]


# 环境初始化
grid_map = DummyGridMap(20, 20)
s_start = (2, 2)
s_goal = (18, 18)
gs_pos = (0, 0)
jammer = DummyJammer(10, 10)  # 干扰源初始在地图正中间

# ================= 2. 初始化“黄金 5 阵型” =================
# 特别注意：Standard D* Lite 是通过把 DStarComm 的 alpha 设为 0 来实现的！
standard_dstar = DStarComm(s_start, s_goal)
standard_dstar.alpha = 0.0  # 切断通信感知，退化为瞎子

proposed_dstar = DStarComm(s_start, s_goal)
proposed_dstar.alpha = 10.0  # 开启强大的跨层通信感知

planners = {
    "1. Pure A* (几何基线)": AStar(s_start, s_goal),
    "2. Standard D* Lite (动态基线)": standard_dstar,
    "3. A*-Comm (通信重算基线)": AStarComm(s_start, s_goal),
    "4. CA-RRT* (前沿采样基线)": CARRTStar(s_start, s_goal),
    "5. D* Lite-Comm (本文算法)": proposed_dstar
}

# ================= 3. 静态规划测试 (初次化) =================
print("========== 阶段 1: 初次静态规划测试 ==========")
for name, planner in planners.items():
    try:
        t0 = time.time()
        path = planner.plan(grid_map, gs=gs_pos, jammers=[jammer])
        t1 = time.time()

        path_len = len(path) if path else 0
        print(f"[{name}] 初次规划成功 | 耗时: {(t1 - t0) * 1000:>6.2f} ms | 路径点数: {path_len}")
    except Exception as e:
        print(f"[{name}] ❌ plan() 崩溃! 原因: {e}")

# ================= 4. 动态重规划测试 (移动干扰源) =================
print("\n========== 阶段 2: 动态重规划测试 (干扰源移动) ==========")
# 把干扰源往必经之路上移动，逼迫通信感知算法绕路
jammer.pos = [14, 14]
print("⚠️ 干扰源已突发移动至 (14, 14). 开始触发所有算法的 dynamic_replan()...\n")

for name, planner in planners.items():
    try:
        t0 = time.time()
        # 对于 A* 等没有增量更新能力的算法，我们在 base_planner 里设定了它会直接调用 plan() 重算
        if hasattr(planner, 'dynamic_replan'):
            path = planner.dynamic_replan()
        else:
            path = planner.plan(grid_map, gs=gs_pos, jammers=[jammer])
        t1 = time.time()

        path_len = len(path) if path else 0
        print(f"[{name}] 重规划成功 | 耗时: {(t1 - t0) * 1000:>6.2f} ms | 路径点数: {path_len}")

    except Exception as e:
        print(f"[{name}] ❌ dynamic_replan() 崩溃! 原因: {e}")

print("\n🎉 沙盒代码测试完毕！")