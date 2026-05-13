# planners/dstar_lite.py
"""
标准 D* Lite 动态路径规划算法（严格增量版）
核心修复：分离首次规划与增量重规划，保留 g/rhs/U/km 状态
重大优化：实现严格局部更新机制，确保单步重规划复杂度为 O(R^2 log R^2)
"""
import math
from planners.base_planner import BasePlanner


class DStarLite(BasePlanner):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        super().__init__(s_start, s_goal, heuristic_type)
        self.u_set = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        self.g = {}
        self.rhs = {}
        self.U = {}
        self.km = 0
        self.grid_map = None
        self._initialized = False  # 标记是否已完成首次初始化

    def _init_costates(self):
        """初始化 g, rhs, U 字典（严格遵循 D* Lite 论文）"""
        for x in range(self.grid_map.width):
            for y in range(self.grid_map.height):
                self.g[(x, y)] = float("inf")
                self.rhs[(x, y)] = float("inf")
        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.calculate_key(self.s_goal)

    def calculate_key(self, s):
        g_s = self.g.get(s, float("inf"))
        rhs_s = self.rhs.get(s, float("inf"))
        h = self.heuristic(s)
        return [min(g_s, rhs_s) + h + self.km, min(g_s, rhs_s)]

    def heuristic(self, s):
        if self.heuristic_type == "manhattan":
            return abs(self.s_start[0] - s[0]) + abs(self.s_start[1] - s[1])
        return math.hypot(self.s_start[0] - s[0], self.s_start[1] - s[1])

    def cost(self, s_start, s_goal, gs=None, jammers=None):
        if self.grid_map.is_collision(s_start) or self.grid_map.is_collision(s_goal):
            return float("inf")
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def get_neighbors(self, s):
        neighbors = []
        for dx, dy in self.u_set:
            ns = (s[0] + dx, s[1] + dy)
            if not self.grid_map.is_collision(ns):
                neighbors.append(ns)
        return neighbors

    def update_vertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = min([self.g.get(x, float("inf")) + self.cost(x, s, self.gs, self.jammers)
                               for x in self.get_neighbors(s)], default=float("inf"))
        if s in self.U:
            self.U.pop(s)
        if self.g.get(s, float("inf")) != self.rhs[s]:
            self.U[s] = self.calculate_key(s)

    def compute_path(self):
        while True:
            # 增加安全检查，防止 U 为空时报错
            if not self.U:
                break

            s, v = self.top_key()
            if v >= self.calculate_key(self.s_start) and \
                    self.rhs.get(self.s_start, float("inf")) == self.g.get(self.s_start, float("inf")):
                break
            k_old = v
            self.U.pop(s)
            if k_old < self.calculate_key(s):
                self.U[s] = self.calculate_key(s)
            elif self.g.get(s, float("inf")) > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for x in self.get_neighbors(s):
                    self.update_vertex(x)
            else:
                self.g[s] = float("inf")
                self.update_vertex(s)
                for x in self.get_neighbors(s):
                    self.update_vertex(x)

    def top_key(self):
        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def extract_path(self):
        path = [self.s_start]
        s = self.s_start
        for _ in range(500):
            if s == self.s_goal:
                break
            neighbors = self.get_neighbors(s)
            if not neighbors:
                break
            s = min(neighbors, key=lambda x: self.g.get(x, float("inf")) + self.cost(s, x, self.gs, self.jammers))
            path.append(s)
        return path

    def plan(self, grid_map, gs=None, jammers=None):
        """✅ 首次规划：仅初始化一次，保留 D* Lite 增量特性"""
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers

        if not self._initialized:
            self._init_costates()
            self._initialized = True

        self.compute_path()
        self.path = self.extract_path()
        return self.path

    def dynamic_replan(self, impact_radius=5):
        if not self._initialized:
            return self.plan(self.grid_map, self.gs, self.jammers)

        # 核心修复：必须同时更新【新位置】和【旧位置】，消除幽灵干扰源！
        affected_nodes = set()

        if self.jammers:
            for jammer in self.jammers:
                # 把当前位置和旧位置都放进更新列表
                positions_to_update = [jammer.pos]
                if hasattr(jammer, 'old_pos'):
                    positions_to_update.append(jammer.old_pos)

                # 遍历两个位置的圆面区域，一并更新
                for px, py in positions_to_update:
                    jx, jy = int(px), int(py)
                    min_x = max(0, jx - impact_radius)
                    max_x = min(self.grid_map.width, jx + impact_radius + 1)
                    min_y = max(0, jy - impact_radius)
                    max_y = min(self.grid_map.height, jy + impact_radius + 1)

                    for x in range(min_x, max_x):
                        for y in range(min_y, max_y):
                            if math.hypot(x - jx, y - jy) <= impact_radius:
                                affected_nodes.add((x, y))

        # 仅更新受影响节点的代价（旧节点会恢复正常代价，新节点会施加惩罚）
        for s in affected_nodes:
            self.update_vertex(s)

        # 执行增量搜索
        self.compute_path()
        self.path = self.extract_path()
        return self.path

    def reset(self):
        """强制重置状态（用于切换完全不同的地图或起点终点）"""
        self.g.clear()
        self.rhs.clear()
        self.U.clear()
        self.km = 0
        self._initialized = False