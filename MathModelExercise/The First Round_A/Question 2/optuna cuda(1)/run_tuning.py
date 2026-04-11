# -*- coding: utf-8 -*-
"""
Optuna调参主程序 - PyTorch CUDA加速版（修复版）
"""

import numpy as np
import random
import time
from multiprocessing import Pool, cpu_count
import json
from datetime import datetime
from pathlib import Path
import logging
import optuna
import traceback
import torch

# 导入模拟器
from uav_simulator import UAVSimulatorCUDA, max_steps, dt, M, N, CUDA_AVAILABLE
import config

# 设置多进程启动方式（Windows需要）
if __name__ == '__main__':
    from multiprocessing import set_start_method

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass


# ======================== 设置日志 ========================
def setup_logging(role="master"):
    log_dir = Path(f"logs_{role}")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{role}_{timestamp}.log"

    # 清除已有的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return log_file, timestamp


# ======================== 评估函数 ========================
def evaluate_params(params, K, N_sim, n_processes=1):
    """评估一组参数"""
    random.seed(42)
    np.random.seed(42)

    # 生成初始位置（1-based）
    init_positions = [(random.randint(1, M), random.randint(1, N)) for _ in range(N_sim)]
    uav_start = [(1, 1)] * K

    # 添加use_pso标志
    params['use_pso'] = True

    # 创建模拟器
    simulator = UAVSimulatorCUDA()

    # 批量仿真（使用单进程避免CUDA冲突）
    detect_steps = []
    for init_pos in init_positions:
        step = simulator.run_simulation(K, init_pos, uav_start, params)
        detect_steps.append(step)

    detect_steps = np.array(detect_steps)
    avg_step = np.mean(detect_steps)
    avg_time = avg_step * dt

    time_10h_steps = 10.0 / dt
    rate_10h = np.sum(detect_steps < time_10h_steps) / N_sim * 100
    rate_40h = np.sum(detect_steps < max_steps) / N_sim * 100

    # 清理GPU内存
    del simulator
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()

    return {
        'avg_time_hours': avg_time,
        'detection_rate_10h': rate_10h,
        'detection_rate_40h': rate_40h,
        'all_steps': detect_steps.tolist()
    }


# ======================== Optuna目标函数 ========================
class Objective:
    def __init__(self, K, N_sim, role):
        self.K = K
        self.N_sim = N_sim
        self.role = role
        self.trial_results = []
        self.best_so_far = float('inf')

    def __call__(self, trial):
        # 从搜索空间采样
        params = {}
        for param_name, param_config in config.SEARCH_SPACE.items():
            try:
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            except Exception as e:
                logging.warning(f"参数 {param_name} 采样失败: {e}")
                params[param_name] = param_config['low']

        # 确保 w_end < w_start
        if params.get('w_end', 0) >= params.get('w_start', 1):
            params['w_end'] = params['w_start'] - 0.1

        logging.info(f"[{self.role}] Trial {trial.number}: 测试参数 {params}")
        start_time = time.time()

        # 评估参数
        results = evaluate_params(
            params, self.K, self.N_sim, n_processes=1
        )

        elapsed = time.time() - start_time

        # 记录结果
        trial.set_user_attr('detection_rate_10h', results['detection_rate_10h'])
        trial.set_user_attr('detection_rate_40h', results['detection_rate_40h'])
        trial.set_user_attr('elapsed_seconds', elapsed)

        logging.info(f"[{self.role}] Trial {trial.number}: "
                     f"时间={results['avg_time_hours']:.2f}h, "
                     f"10h率={results['detection_rate_10h']:.1f}%, "
                     f"耗时={elapsed:.1f}s")

        self.trial_results.append({
            'trial_number': trial.number,
            'params': params,
            'results': results,
            'elapsed': elapsed
        })

        # 更新最佳值
        if results['avg_time_hours'] < self.best_so_far:
            self.best_so_far = results['avg_time_hours']
            logging.info(f"[{self.role}] 新的最佳值: {self.best_so_far:.2f}h")

        return results['avg_time_hours']


# ======================== 主函数 ========================
def main():
    # 获取角色
    role = getattr(config, 'ROLE', 'master')

    print(f"\n{'=' * 60}")
    print(f"PyTorch CUDA加速调参 - 角色: {role}")
    print(f"CUDA可用: {CUDA_AVAILABLE}")
    print(f"{'=' * 60}\n")

    # 设置日志
    log_file, timestamp = setup_logging(role)

    # 测试模拟器
    logging.info("测试模拟器...")
    try:
        simulator = UAVSimulatorCUDA()
        simulator.print_device_info()

        # 快速测试
        test_params = {
            'H': 6,
            'pop_size': 12,
            'max_iter': 20,
            'w_start': 0.9,
            'w_end': 0.4,
            'c1': 1.5,
            'c2': 1.5,
            'penalty_coef': 1000.0,
            'use_pso': True
        }
        uav_start = [(1, 1)] * config.FIXED_PARAMS['K']
        init_pos = (random.randint(1, M), random.randint(1, N))
        test_result = simulator.run_simulation(config.FIXED_PARAMS['K'], init_pos, uav_start, test_params)
        logging.info(f"测试仿真成功，发现步数: {test_result}")

    except Exception as e:
        logging.error(f"模拟器测试失败: {e}")
        traceback.print_exc()
        return

    # 创建结果目录
    result_dir = Path(f"results_{role}")
    result_dir.mkdir(exist_ok=True)

    # 创建Optuna study
    study_name = f"uav_search_{role}_{timestamp}"
    storage_url = f"sqlite:///{result_dir}/study_{role}.db"

    try:
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage_url,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
    except Exception as e:
        logging.warning(f"创建study失败，使用内存存储: {e}")
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )

    # 创建目标函数
    objective = Objective(
        K=config.FIXED_PARAMS['K'],
        N_sim=config.N_SIM_PER_EVAL,
        role=role
    )

    # 运行优化
    logging.info(f"开始优化 - 试验次数: {config.N_TRIALS}")
    try:
        study.optimize(
            objective,
            n_trials=config.N_TRIALS,
            timeout=config.OPTUNA_CONFIG.get('timeout', None),
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logging.info("用户中断优化")
    except Exception as e:
        logging.error(f"优化过程出错: {e}")
        traceback.print_exc()

    # 保存结果
    if study.best_trial:
        save_results(study, objective.trial_results, role, timestamp, result_dir)
        print_best_results(study, role)
    else:
        logging.error("没有成功的试验结果")

    logging.info(f"\n{role} 调参完成！")


def save_results(study, trial_results, role, timestamp, result_dir):
    """保存结果"""
    # 保存最佳参数
    best_config_file = result_dir / f"best_config_{timestamp}.py"
    with open(best_config_file, 'w', encoding='utf-8') as f:
        f.write(f"# 最佳参数配置 - {role}\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                f.write(f"{param} = {value}\n")
            else:
                f.write(f"{param} = {value}\n")
        f.write(f"\n# 最佳平均发现时间: {study.best_value:.2f} 小时\n")

        # 找到对应的试验结果
        for trial in trial_results:
            if trial['trial_number'] == study.best_trial.number:
                f.write(f"# 10小时发现率: {trial['results']['detection_rate_10h']:.1f}%\n")
                f.write(f"# 40小时发现率: {trial['results']['detection_rate_40h']:.1f}%\n")
                break

    # 保存完整结果
    results = {
        'role': role,
        'timestamp': timestamp,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'trials': trial_results,
        'search_space': config.SEARCH_SPACE
    }

    result_file = result_dir / f"results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"结果已保存到: {result_dir}")


def print_best_results(study, role):
    """打印最佳结果"""
    logging.info("\n" + "=" * 60)
    logging.info(f"[{role}] 调参完成！最佳结果")
    logging.info("=" * 60)
    logging.info(f"最佳平均发现时间: {study.best_value:.2f} 小时")
    logging.info("\n最佳参数组合:")
    for param, value in study.best_params.items():
        logging.info(f"  {param} = {value}")


if __name__ == "__main__":
    main()