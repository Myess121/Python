# swarm/swarm_planner.py
"""
多无人机协同规划器
实现：任务分配 + 独立通信感知规划 + 冲突检测
"""
from planners.dstar_comm import DStarComm
from swarm.task_assignment import assign_tasks  # 调用外部任务分配模块


class SwarmPlanner:
    def __init__(self, uavs, grid_map, gs, jammers):
        self.uavs = uavs
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers

    def plan_collaboratively(self):
        """
        协同规划主流程
        1. 任务分配（区域划分，避开低SNR层）
        2. 独立规划（每架无人机运行 D* Comm）
        3. 冲突检测（时空路径交叉分析）
        """
        # 步骤 1：任务分配
        self.uavs = assign_tasks(self.uavs, self.grid_map, self.jammers)

        # 步骤 2：独立规划
        all_paths = {}
        for uav in self.uavs:
            planner = DStarComm(uav.start_pos, uav.goal_pos)
            path = planner.plan(self.grid_map, self.gs, self.jammers)
            uav.path = path
            all_paths[uav.id] = path

        # 步骤 3：冲突检测
        self._check_conflicts(all_paths)

        return all_paths

    def _check_conflicts(self, all_paths):
        """检测路径交叉点（用于论文防碰撞分析）"""
        conflicts = []
        uav_ids = list(all_paths.keys())
        for i in range(len(uav_ids)):
            for j in range(i + 1, len(uav_ids)):
                path1 = all_paths[uav_ids[i]]
                path2 = all_paths[uav_ids[j]]
                # 检查同一步数是否占据同一网格
                for t in range(min(len(path1), len(path2))):
                    if path1[t] == path2[t]:
                        conflicts.append((uav_ids[i], uav_ids[j], t, path1[t]))

        if conflicts:
            print(f"⚠️ 检测到 {len(conflicts)} 个时空冲突点（论文可说明采用时间窗调度或高度层分离解决）")
        else:
            print("✅ 路径无时空冲突")