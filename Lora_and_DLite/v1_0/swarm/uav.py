# swarm/uav.py
"""
无人机类：封装单机属性、状态和通信模型
"""
import math


class UAV:
    def __init__(self, uav_id, start_pos, goal_pos, role="scout"):
        self.id = uav_id
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.role = role  # "scout", "relay", "leader"

        # 飞行状态
        self.current_pos = start_pos
        self.path = []
        self.path_index = 0

        # 通信参数（LoRa 模块）
        self.tx_power = 14.0  # dBm
        self.snr_history = []

    def move_to_next_waypoint(self):
        """移动到路径的下一个航点"""
        if self.path_index < len(self.path) - 1:
            self.path_index += 1
            self.current_pos = self.path[self.path_index]
            return True
        return False

    def get_communication_range(self, other_uav):
        """计算与另一架无人机的通信质量（简化模型）"""
        d = math.hypot(self.current_pos[0] - other_uav.current_pos[0],
                       self.current_pos[1] - other_uav.current_pos[1])
        # 简化：距离越远，SNR 越低
        snr = 20 - 10 * math.log10(d + 1)
        return snr

    def reset(self):
        """重置状态（用于多次实验）"""
        self.current_pos = self.start_pos
        self.path = []
        self.path_index = 0
        self.snr_history = []