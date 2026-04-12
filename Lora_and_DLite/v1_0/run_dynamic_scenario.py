# run_dynamic_scenario.py
"""
动态干扰场景测试脚本
模拟干扰源移动，验证通信感知 D* Lite 的实时重规划能力（严格增量版）
"""
from environment import GridMap, GroundStation, Jammer
from planners.dstar_comm import DStarComm
from config import START_POS, GOAL_POS
import utils

def main():
    print("🚀 开始动态干扰场景模拟（严格增量版）...")

    grid_map = GridMap()
    gs = GroundStation()
    jammer = Jammer(pos=(10.0, 10.0), velocity=(1.5, 0.3))

    # ✅ 全局唯一实例（不重置 g/rhs/U/km 状态）
    planner = DStarComm(START_POS, GOAL_POS)

    frames_data = []

    # 1. 首次规划
    path = planner.plan(grid_map, gs, [jammer])
    frames_data.append({"time": 0, "path": path, "jammer_pos": jammer.pos.copy()})

    # 2. 动态循环：干扰源移动 -> 增量重规划
    for t in range(1, 6):
        jammer.update()
        path = planner.dynamic_replan()  # ✅ 调用增量重规划接口
        frames_data.append({"time": t, "path": path, "jammer_pos": jammer.pos.copy()})
        print(f"⏳ Step {t}: Jammer@{jammer.pos}, PathLen={len(path)}")

    # 3. 生成演化图（只调用一次，utils.py 已配置 plt.close() 不会阻塞）
    utils.plot_dynamic_evolution(frames_data, grid_map)
    print("✅ 模拟结束！")

if __name__ == "__main__":
    main()