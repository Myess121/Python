# planners/ca_rrt_star.py
"""
通信感知快速扩展随机树 (CA-RRT*)
经典的高级 Baseline 算法 (基于采样)。
用于衬托 D* Lite 算法在动态干扰下重规划的高效性、确定性以及零方差优势。
"""
import math
import random
from planners.base_planner import BasePlanner
from environment import calculate_physical_sinr, get_normalized_comm_score
from config import COMM_WEIGHT


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # 从起点到当前节点的累计代价 (距离 + 通信惩罚)


class CARRTStar(BasePlanner):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        super().__init__(s_start, s_goal, heuristic_type)
        self.alpha = COMM_WEIGHT
        self.replan_score_th = 0.8

        # RRT* 专属超参数 (您可以根据地图大小进行微调)
        self.max_iter = 5000  # 最大采样次数
        self.step_size = 3.0  # 树枝生长的步长
        self.search_radius = 5.0  # 邻近节点重连半径
        self.goal_sample_rate = 0.1  # 10%的概率直接向终点采样，加速收敛

    def plan(self, grid_map, gs=None, jammers=None):
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers

        # 树的初始化
        self.node_list = [Node(self.s_start[0], self.s_start[1])]

        for i in range(self.max_iter):
            # 1. 随机采样
            rnd_node = self.get_random_node()

            # 2. 找树上最近的节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 3. 沿方向生长一个 step_size
            new_node = self.steer(nearest_node, rnd_node, self.step_size)

            # 4. 碰撞检测 (包含几何障碍物)
            if self.check_collision(nearest_node, new_node):

                # 5. RRT* 核心：寻找范围内的邻近节点 (Near Nodes)
                near_inds = self.find_near_nodes(new_node)

                # 6. 为新节点选择最优父节点 (考虑距离 + 通信代价)
                new_node = self.choose_parent(new_node, near_inds)

                if new_node:
                    self.node_list.append(new_node)
                    # 7. 树枝重连 (Rewiring)，优化已有路径
                    self.rewire(new_node, near_inds)

        # 尝试生成最终路径
        return self.generate_final_course()

    def dynamic_replan(self):
        """
        RRT* 面对动态干扰的致命弱点：
        由于整棵树的代价在不断变化，最稳妥（但也最慢）的做法是清空树重新采样。
        这正是用来衬托 D* Lite 增量更新优势的关键！
        """
        return self.plan(self.grid_map, self.gs, self.jammers)

    def cost(self, n1, n2):
        """计算两点之间的单步代价：欧式距离 + 通信惩罚"""
        dist_cost = math.hypot(n1.x - n2.x, n1.y - n2.y)

        comm_cost = 0.0
        if self.gs is not None and self.jammers is not None:
            # RRT* 是连续空间的，为了加速，我们近似计算新节点 n2 处的通信质量
            pos = (n2.x, n2.y)
            sinr = calculate_physical_sinr(pos, self.gs, self.jammers)
            comm_score = get_normalized_comm_score(sinr)

            if comm_score < self.replan_score_th:
                comm_cost = self.alpha * (self.replan_score_th - comm_score) * dist_cost  # 乘以距离作为路径积分近似

        return dist_cost + comm_cost

    # ================= 以下为 RRT* 底层几何操作库 =================

    def get_random_node(self):
        """带目标偏置的随机采样"""
        if random.random() > self.goal_sample_rate:
            x = random.uniform(0, self.grid_map.width - 1)
            y = random.uniform(0, self.grid_map.height - 1)
        else:
            x, y = self.s_goal[0], self.s_goal[1]
        return Node(x, y)

    def steer(self, from_node, to_node, extend_length):
        """向目标点生长"""
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # 连续插值，用于碰撞检测
        n_expand = math.floor(extend_length / 0.5)
        for _ in range(n_expand):
            new_node.x += 0.5 * math.cos(theta)
            new_node.y += 0.5 * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= 0.5:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        return new_node

    def check_collision(self, node1, node2):
        """沿轨迹进行连续空间的网格碰撞检测"""
        for ix, iy in zip(node2.path_x, node2.path_y):
            # 将连续坐标转为网格整数坐标
            grid_pos = (int(round(ix)), int(round(iy)))
            # 边界检查
            if not (0 <= grid_pos[0] < self.grid_map.width and 0 <= grid_pos[1] < self.grid_map.height):
                return False
            # 障碍物检查
            if self.grid_map.is_collision(grid_pos):
                return False
        return True

    def find_near_nodes(self, new_node):
        """在半径内寻找邻近节点"""
        nnode = len(self.node_list) + 1
        r = self.search_radius * math.sqrt((math.log(nnode) / nnode))
        r = min(r, self.step_size * 2.0)
        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        """考虑通信代价，重选最优父节点"""
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node, self.step_size)
            if t_node and self.check_collision(near_node, t_node):
                # 累加：父节点累计代价 + 边代价 (包含通信惩罚)
                costs.append(near_node.cost + self.cost(near_node, t_node))
            else:
                costs.append(float("inf"))

        min_cost = min(costs)
        if min_cost == float("inf"):
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node, self.step_size)
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        """优化局部树结构"""
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node, self.step_size)
            if not edge_node:
                continue
            edge_node.cost = new_node.cost + self.cost(new_node, near_node)

            if near_node.cost > edge_node.cost:
                if self.check_collision(new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = edge_node.cost
                    self.propagate_cost_to_leaves(new_node)

    def propagate_cost_to_leaves(self, parent_node):
        """递归更新子节点代价"""
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost + self.cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def generate_final_course(self):
        """反向回溯生成路径，并将其转换为整数网格坐标序列，以便与 D* Lite 统一格式"""
        dlist = [(node.x - self.s_goal[0]) ** 2 + (node.y - self.s_goal[1]) ** 2 for node in self.node_list]
        goal_inds = [dlist.index(i) for i in dlist if i <= self.step_size ** 2]

        if not goal_inds:
            return []  # 找不到路径

        min_cost = float("inf")
        goal_ind = None
        for i in goal_inds:
            if self.node_list[i].cost < min_cost:
                min_cost = self.node_list[i].cost
                goal_ind = i

        node = self.node_list[goal_ind]
        path = [(self.s_goal[0], self.s_goal[1])]
        while node.parent is not None:
            # 转换为整型坐标
            path.append((int(round(node.x)), int(round(node.y))))
            node = node.parent
        path.append((self.s_start[0], self.s_start[1]))
        path.reverse()

        # 去除连续重复的坐标点
        cleaned_path = [path[0]]
        for p in path[1:]:
            if p != cleaned_path[-1]:
                cleaned_path.append(p)

        return cleaned_path