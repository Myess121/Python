# planners/base_planner.py
import math
from abc import ABC, abstractmethod

class BasePlanner(ABC):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.path = []
        self.grid_map = None
        self.gs = None
        self.jammers = None

    @abstractmethod
    def plan(self, grid_map, gs=None, jammers=None):
        """统一规划接口，子类必须实现"""
        pass

    @abstractmethod
    def cost(self, s_start, s_goal, gs=None, jammers=None):
        """统一代价接口，子类可重写（如加入通信代价）"""
        pass

    def heuristic(self, s):
        """启发式函数"""
        if self.heuristic_type == "manhattan":
            return abs(self.s_start[0] - s[0]) + abs(self.s_start[1] - s[1])
        return math.hypot(self.s_start[0] - s[0], self.s_start[1] - s[1])