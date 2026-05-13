# swarm/task_assignment.py
"""
任务分配模块：为多无人机分配目标区域，避免路径冲突，兼顾通信质量
"""
from environment import calculate_snr, GroundStation
from config import *
import math


def assign_tasks(uavs, grid_map, jammers):
    """
    任务分配策略：基于干扰源位置和 Y 轴分层分配
    1. 划分 Y 轴区域层，计算每层平均 SNR
    2. 按 UAV 初始 Y 坐标分配对应层，优先高 SNR 层
    3. 低 SNR 层微调目标 Y 坐标，避开干扰
    """
    num_uavs = len(uavs)
    if num_uavs == 0:
        return uavs

    # 步骤 1：划分 Y 层（根据无人机数量和地图高度）
    layer_height = MAP_HEIGHT // num_uavs  # 每层高度

    # 步骤 2：计算每层的平均 SNR（采样中间点）
    layer_snrs = []
    ground_station = GroundStation()  # 地面站实例

    for i in range(num_uavs):
        y_layer = (i + 0.5) * layer_height  # 层中间位置（避免边界）
        # 采样该层的多个 X 点计算 SNR
        snr_samples = []
        for x in range(0, MAP_WIDTH, 5):  # 每隔 5 个 X 采样，平衡效率和覆盖
            pos = (x, y_layer)
            if not grid_map.is_collision(pos):  # 避开障碍物
                snr = calculate_snr(pos, ground_station, jammers)
                snr_samples.append(snr)
        # 计算平均 SNR（无有效样本则设为 0）
        avg_snr = sum(snr_samples) / len(snr_samples) if snr_samples else 0
        layer_snrs.append(avg_snr)

    # 步骤 3：按初始 Y 坐标排序 UAV，分配对应层
    sorted_uavs = sorted(uavs, key=lambda u: u.start_pos[1])

    for i, uav in enumerate(sorted_uavs):
        target_y = (i + 0.5) * layer_height  # 初始目标 Y（层中间）

        # 若该层 SNR 低于阈值，微调目标 Y 避开干扰
        if layer_snrs[i] < SNR_THRESHOLD:
            # 向上/下微调 2 格（根据层位置选择方向）
            if i < num_uavs // 2:
                target_y += 2  # 上层无人机向上微调
            else:
                target_y -= 2  # 下层无人机向下微调
            # 确保 Y 坐标不越界
            target_y = max(0, min(target_y, MAP_HEIGHT - 1))

        # 更新 UAV 目标位置（保持 X 目标不变，仅调整 Y）
        uav.goal_pos = (uav.goal_pos[0], target_y)

    return uavs


def test_task_assignment():
    """测试任务分配逻辑"""
    from environment import GridMap, Jammer
    from swarm.uav import UAV

    # 初始化环境
    grid_map = GridMap()
    jammers = [Jammer(pos=(25.0, 15.0), velocity=(0.5, 0))]  # 干扰源在中间层
    ground_station = GroundStation()

    # 创建 3 架初始目标 Y 相同的无人机（测试分配调整）
    uavs = [
        UAV(0, start_pos=(5, 5), goal_pos=(45, 15)),
        UAV(1, start_pos=(5, 15), goal_pos=(45, 15)),
        UAV(2, start_pos=(5, 25), goal_pos=(45, 15))
    ]

    print("🔍 分配前目标 Y 坐标:")
    for uav in uavs:
        print(f"  UAV {uav.id}: 目标 Y={uav.goal_pos[1]}")

    # 执行任务分配
    assigned_uavs = assign_tasks(uavs, grid_map, jammers)

    print("\n🔍 分配后目标 Y 坐标:")
    for uav in assigned_uavs:
        print(f"  UAV {uav.id}: 目标 Y={uav.goal_pos[1]}")

    # 验证 SNR 改善（对比分配前后的路径中点 SNR）
    print("\n📊 路径中点 SNR 对比:")
    for uav in assigned_uavs:
        mid_pos = (25, uav.goal_pos[1])  # 路径中点（X=25）
        snr_before = calculate_snr((25, 15), ground_station, jammers)  # 原目标 Y=15 的 SNR
        snr_after = calculate_snr(mid_pos, ground_station, jammers)
        print(f"  UAV {uav.id}: 原 SNR={snr_before:.2f} dB -> 新 SNR={snr_after:.2f} dB")

    print("✅ 任务分配测试完成！")


if __name__ == "__main__":
    test_task_assignment()