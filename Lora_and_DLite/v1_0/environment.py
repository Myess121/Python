# environment.py
"""
环境建模模块
包含：网格地图、地面站通信模型、动态干扰源、SNR计算核心函数
"""
import math
from config import *


class GridMap:
    """网格地图与障碍物管理"""

    def __init__(self):
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT
        self.obstacles = set(OBSTACLES)

    def is_collision(self, pos):
        """检测位置是否碰撞（越界或进入障碍物）"""
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return pos in self.obstacles

    def get_neighbors(self, pos):
        """获取8邻域可行节点（用于路径规划）"""
        x, y = pos
        neighbors = []
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not self.is_collision((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors


class GroundStation:
    """地面站通信模型（有用信号源）"""

    def __init__(self):
        self.pos = GS_POS
        self.tx_power = GS_TX_POWER

    def get_signal_strength(self, uav_pos):
        """计算无人机接收到的地面站信号强度 (RSSI, dBm)"""
        d = math.hypot(uav_pos[0] - self.pos[0], uav_pos[1] - self.pos[1])
        d = max(d, D0)  # 避免距离过近导致对数计算异常

        # 对数距离路径损耗模型: PL(d) = PL_d0 + 10*n*log10(d/d0)
        path_loss = PL_D0 + 10 * N_FACTOR * math.log10(d / D0)
        rssi = self.tx_power - path_loss
        return rssi

# environment.py 中找到 class Jammer 并替换为：
class Jammer:
    """动态干扰源模型（支持实验时自定义位置/速度）"""
    def __init__(self, pos=None, power=None, velocity=None):
        from config import JAMMER_INIT_POS, JAMMER_POWER, JAMMER_VELOCITY
        self.pos = list(pos if pos is not None else JAMMER_INIT_POS)
        self.power = power if power is not None else JAMMER_POWER
        self.velocity = list(velocity if velocity is not None else JAMMER_VELOCITY)

    def get_interference(self, uav_pos):
        """计算干扰强度"""
        d = math.hypot(uav_pos[0] - self.pos[0], uav_pos[1] - self.pos[1])
        d = max(d, D0)
        path_loss_jammer = PL_D0 + 10 * N_FACTOR * math.log10(d / D0)
        return self.power - path_loss_jammer

    def update(self):
        """移动干扰源 + 边界反弹"""
        from config import MAP_WIDTH, MAP_HEIGHT
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]

        if self.pos[0] <= 0 or self.pos[0] >= MAP_WIDTH - 1:
            self.velocity[0] *= -1
        if self.pos[1] <= 0 or self.pos[1] >= MAP_HEIGHT - 1:
            self.velocity[1] *= -1

'''class Jammer:
    """动态干扰源模型"""

    def __init__(self):
        self.pos = list(JAMMER_INIT_POS)  # 使用list保证可修改
        self.power = JAMMER_POWER
        self.velocity = list(JAMMER_VELOCITY)

    def get_interference(self, uav_pos):
        """计算干扰源对无人机的干扰强度 (dBm)"""
        d = math.hypot(uav_pos[0] - self.pos[0], uav_pos[1] - self.pos[1])
        d = max(d, D0)

        # 干扰信号同样遵循路径损耗模型
        path_loss_jammer = PL_D0 + 10 * N_FACTOR * math.log10(d / D0)
        interference = self.power - path_loss_jammer
        return interference

    def update(self):
        """更新干扰源位置：匀速移动 + 边界反弹"""
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]

        # 边界检测与速度反向
        if self.pos[0] <= 0 or self.pos[0] >= MAP_WIDTH - 1:
            self.velocity[0] *= -1
        if self.pos[1] <= 0 or self.pos[1] >= MAP_HEIGHT - 1:
            self.velocity[1] *= -1'''


def calculate_snr(uav_pos, ground_station, jammers):
    """
    计算信噪比 SNR (dB)
    工程简化公式: SNR ≈ RSSI - I_total - N (适用于路径规划实时计算)

    参数:
        uav_pos: 无人机坐标 (x, y)
        ground_station: GroundStation 实例
        jammers: Jammer 实例列表 (支持单/多干扰源)
    返回:
        SNR值 (dB)
    """
    # 1. 地面站有用信号
    signal = ground_station.get_signal_strength(uav_pos)

    # 2. 总干扰强度
    total_interference = sum(jammer.get_interference(uav_pos) for jammer in jammers)

    # 3. SNR计算 (dB域)
    snr = signal - total_interference - NOISE_POWER
    return snr