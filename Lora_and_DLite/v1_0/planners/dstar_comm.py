# planners/dstar_comm.py
import math
from planners.dstar_lite import DStarLite
from environment import calculate_snr
from config import COMM_WEIGHT, SNR_THRESHOLD


class DStarComm(DStarLite):
    def __init__(self, s_start, s_goal, heuristic_type="euclidean"):
        super().__init__(s_start, s_goal, heuristic_type)
        self.comm_weight = COMM_WEIGHT
        self.snr_threshold = SNR_THRESHOLD

    def cost(self, s_start, s_goal, gs=None, jammers=None):
        if self.grid_map.is_collision(s_start) or self.grid_map.is_collision(s_goal):
            return float("inf")
        dist_cost = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

        if self.gs is not None and self.jammers is not None:
            snr = calculate_snr(s_goal, self.gs, self.jammers)
            if snr < self.snr_threshold:
                comm_cost = self.comm_weight * (self.snr_threshold - snr)
            else:
                comm_cost = 0.0
        else:
            comm_cost = 0.0
        return dist_cost + comm_cost
