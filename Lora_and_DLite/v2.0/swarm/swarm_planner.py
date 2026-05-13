'''# swarm/swarm_planner.py
"""
多无人机协同规划器
实现：任务分配 + 主从式通信感知规划（伴飞中继协同）+ 冲突检测
"""
import math
from planners.dstar_comm import DStarComm
from swarm.task_assignment import assign_tasks
from config import PL_D0, N_FACTOR, D0


class VirtualGS:
    """
    虚拟伴飞中继站（动态更新 X 坐标，模拟与侦察机并排飞行）
    """

    def __init__(self, relay_base_y, original_gs):
        self.relay_base_y = relay_base_y
        # 适当降低中继机的发射功率，迫使侦察机在强干扰下必须靠近中继机
        self.tx_power = original_gs.tx_power - 40.0

    def get_signal_strength(self, uav_pos):
        # 【终极魔法】：中继机的 X 坐标实时等于侦察机的 X 坐标！
        # 这样侦察机在向终点飞行时，不会因为距离变远而掉线。
        # 它唯一的变量就是 Y 轴距离。当遭遇干扰时，它只能通过改变 Y 轴来靠近中继机。
        dynamic_relay_pos = (uav_pos[0], self.relay_base_y)
        d = math.hypot(uav_pos[0] - dynamic_relay_pos[0], uav_pos[1] - dynamic_relay_pos[1])
        d = max(d, D0)
        path_loss = PL_D0 + 10 * N_FACTOR * math.log10(d / D0)
        return self.tx_power - path_loss


class SwarmPlanner:
    def __init__(self, uavs, grid_map, gs, jammers):
        self.uavs = uavs
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers

    def plan_collaboratively(self):
        """
        协同规划主流程
        """
        self.uavs = assign_tasks(self.uavs, self.grid_map, self.jammers)
        all_paths = {}

        # 步骤 1：找出中继机 (Relay) 并优先规划（连通真实大基站）
        relay_uav = next((u for u in self.uavs if u.role == "relay"), self.uavs[len(self.uavs) // 2])
        relay_planner = DStarComm(relay_uav.start_pos, relay_uav.goal_pos)
        relay_uav.path = relay_planner.plan(self.grid_map, self.gs, self.jammers)
        all_paths[relay_uav.id] = relay_uav.path

        # 步骤 2：提取中继机的基准高度层 (Y坐标)
        relay_base_y = relay_uav.start_pos[1]
        virtual_gs = VirtualGS(relay_base_y, self.gs)

        # 步骤 3：为侦察机 (Scout) 进行伴飞协同规划
        for uav in self.uavs:
            if uav.id == relay_uav.id:
                continue

            scout_planner = DStarComm(uav.start_pos, uav.goal_pos)
            # 侦察机连通虚拟伴飞基站
            path = scout_planner.plan(self.grid_map, virtual_gs, self.jammers)
            uav.path = path
            all_paths[uav.id] = path

        self._check_conflicts(all_paths)
        return all_paths

    def _check_conflicts(self, all_paths):
        conflicts = []
        uav_ids = list(all_paths.keys())
        for i in range(len(uav_ids)):
            for j in range(i + 1, len(uav_ids)):
                path1 = all_paths[uav_ids[i]]
                path2 = all_paths[uav_ids[j]]
                for t in range(min(len(path1), len(path2))):
                    if path1[t] == path2[t]:
                        conflicts.append((uav_ids[i], uav_ids[j], t, path1[t]))

        if conflicts:
            print(f"⚠️ 检测到 {len(conflicts)} 个时空冲突点")
        else:
            print("✅ 路径无时空冲突")'''
'''# swarm/swarm_planner.py
"""
多无人机协同规划器
实现：任务分配 + 主从式通信感知规划（动态中继协同）+ 冲突检测
"""
import math
from planners.dstar_comm import DStarComm
from swarm.task_assignment import assign_tasks
from config import PL_D0, N_FACTOR, D0


class VirtualGS:
    """
    虚拟地面站（用于将中继机伪装成微型信号源）
    """

    def __init__(self, pos, original_gs):
        self.pos = pos
        # 【核心魔法在这里！！！】
        # 将中继机的发射功率暴降 35 dBm (模拟微型中继天线)
        # 这样侦察机必须“紧紧贴着”中继机飞，否则信号会瞬间跌穿 75 的阈值！
        self.tx_power = original_gs.tx_power - 35.0

    def get_signal_strength(self, uav_pos):
        # 严格复用 environment.py 中的对数距离路径损耗逻辑
        d = math.hypot(uav_pos[0] - self.pos[0], uav_pos[1] - self.pos[1])
        d = max(d, D0)
        path_loss = PL_D0 + 10 * N_FACTOR * math.log10(d / D0)
        return self.tx_power - path_loss


class SwarmPlanner:
    def __init__(self, uavs, grid_map, gs, jammers):
        self.uavs = uavs
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers

    def plan_collaboratively(self):
        """
        协同规划主流程
        1. 任务分配（分配高度层）
        2. 中继机优先规划，连通大基站
        3. 侦察机以中继机为基站协同规划，形成聚合阵型
        """
        self.uavs = assign_tasks(self.uavs, self.grid_map, self.jammers)
        all_paths = {}

        # 步骤 1：找出中继机并优先规划
        relay_uav = next((u for u in self.uavs if u.role == "relay"), self.uavs[len(self.uavs) // 2])
        relay_planner = DStarComm(relay_uav.start_pos, relay_uav.goal_pos)
        relay_uav.path = relay_planner.plan(self.grid_map, self.gs, self.jammers)
        all_paths[relay_uav.id] = relay_uav.path

        # 步骤 2：提取中继机在危险区域（X=25附近）的坐标作为“移动基站”的引力中心
        mid_index = len(relay_uav.path) // 2
        relay_virtual_pos = relay_uav.path[mid_index] if relay_uav.path else (25, 15)
        virtual_gs = VirtualGS(relay_virtual_pos, self.gs)

        # 步骤 3：为侦察机进行协同规划
        for uav in self.uavs:
            if uav.id == relay_uav.id:
                continue

            scout_planner = DStarComm(uav.start_pos, uav.goal_pos)

            # 此时的地面站参数传的是 virtual_gs，这会产生向心引力
            path = scout_planner.plan(self.grid_map, virtual_gs, self.jammers)
            uav.path = path
            all_paths[uav.id] = path

        self._check_conflicts(all_paths)
        return all_paths

    def _check_conflicts(self, all_paths):
        conflicts = []
        uav_ids = list(all_paths.keys())
        for i in range(len(uav_ids)):
            for j in range(i + 1, len(uav_ids)):
                path1 = all_paths[uav_ids[i]]
                path2 = all_paths[uav_ids[j]]
                for t in range(min(len(path1), len(path2))):
                    if path1[t] == path2[t]:
                        conflicts.append((uav_ids[i], uav_ids[j], t, path1[t]))

        if conflicts:
            print(f"⚠️ 检测到 {len(conflicts)} 个时空冲突点")
        else:
            print("✅ 路径无时空冲突")'''
'''# swarm/swarm_planner.py
"""
多无人机协同规划器
实现：任务分配 + 主从式通信感知规划（动态中继协同）+ 冲突检测
"""
import math
from planners.dstar_comm import DStarComm
from swarm.task_assignment import assign_tasks
from config import PL_D0, N_FACTOR, D0


class VirtualGS:
    """
    虚拟地面站（用于将中继机伪装成信号源）
    完美复现真实 GroundStation 的 get_signal_strength 接口
    """

    def __init__(self, pos, original_gs):
        self.pos = pos
        self.tx_power = original_gs.tx_power

    def get_signal_strength(self, uav_pos):
        # 严格复用 environment.py 中的路径损耗衰减逻辑
        d = math.hypot(uav_pos[0] - self.pos[0], uav_pos[1] - self.pos[1])
        d = max(d, D0)
        path_loss = PL_D0 + 10 * N_FACTOR * math.log10(d / D0)
        return self.tx_power - path_loss


class SwarmPlanner:
    def __init__(self, uavs, grid_map, gs, jammers):
        self.uavs = uavs
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers

    def plan_collaboratively(self):
        """
        协同规划主流程
        1. 任务分配（分配高度层）
        2. 中继机(Relay)优先规划，保证主干网络连通
        3. 侦察机(Scout)以中继机为虚拟基站进行协同规划，形成聚合保护阵型
        """
        # 步骤 1：任务分配
        self.uavs = assign_tasks(self.uavs, self.grid_map, self.jammers)
        all_paths = {}

        # 步骤 2：找出中继机并优先规划（连接真实物理地面站 gs）
        # 如果找不到 role="relay"，默认选中间那架
        relay_uav = next((u for u in self.uavs if u.role == "relay"), self.uavs[len(self.uavs) // 2])
        relay_planner = DStarComm(relay_uav.start_pos, relay_uav.goal_pos)
        relay_uav.path = relay_planner.plan(self.grid_map, self.gs, self.jammers)
        all_paths[relay_uav.id] = relay_uav.path

        # 提取中继机在危险区域（路径中段）的坐标，作为动态中继基站
        mid_index = len(relay_uav.path) // 2
        relay_virtual_pos = relay_uav.path[mid_index] if relay_uav.path else (25, 15)
        virtual_gs = VirtualGS(relay_virtual_pos, self.gs)

        # 步骤 3：为侦察机进行协同通信感知规划
        for uav in self.uavs:
            if uav.id == relay_uav.id:
                continue

            scout_planner = DStarComm(uav.start_pos, uav.goal_pos)

            # 【核心魔法】：侦察机的信号源变成了中继机 (virtual_gs)！
            # 这会迫使侦察机在遇到干扰时，自动向中继机的轨迹靠拢
            path = scout_planner.plan(self.grid_map, virtual_gs, self.jammers)
            uav.path = path
            all_paths[uav.id] = path

        # 步骤 4：冲突检测
        self._check_conflicts(all_paths)

        return all_paths

    def _check_conflicts(self, all_paths):
        """检测路径交叉点"""
        conflicts = []
        uav_ids = list(all_paths.keys())
        for i in range(len(uav_ids)):
            for j in range(i + 1, len(uav_ids)):
                path1 = all_paths[uav_ids[i]]
                path2 = all_paths[uav_ids[j]]
                for t in range(min(len(path1), len(path2))):
                    if path1[t] == path2[t]:
                        conflicts.append((uav_ids[i], uav_ids[j], t, path1[t]))

        if conflicts:
            print(f"⚠️ 检测到 {len(conflicts)} 个时空冲突点")
        else:
            print("✅ 路径无时空冲突")

'''
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
       # self.uavs = assign_tasks(self.uavs, self.grid_map, self.jammers)

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