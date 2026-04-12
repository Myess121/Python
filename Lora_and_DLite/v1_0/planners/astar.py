# planners/astar.py
import math
import heapq
from planners.base_planner import BasePlanner

class AStar(BasePlanner):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        super().__init__(s_start, s_goal, heuristic_type)
        self.moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

    def plan(self, grid_map, gs=None, jammers=None):
        self.grid_map = grid_map
        self.gs = gs
        self.jammers = jammers

        open_set = []
        heapq.heappush(open_set, (0, self.s_start))
        came_from = {self.s_start: None}
        g_score = {self.s_start: 0}
        f_score = {self.s_start: self.heuristic(self.s_start)}
        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == self.s_goal:
                return self._reconstruct_path(came_from, current)

            closed_set.add(current)

            for move in self.moves:
                neighbor = (current[0] + move[0], current[1] + move[1])
                if neighbor in closed_set or self.grid_map.is_collision(neighbor):
                    continue

                tentative_g = g_score[current] + self.cost(current, neighbor, gs, jammers)

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # 无可行路径

    def cost(self, s_start, s_goal, gs=None, jammers=None):
        """A* 仅使用几何距离代价（忽略通信）"""
        if self.grid_map.is_collision(s_start) or self.grid_map.is_collision(s_goal):
            return float('inf')
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path