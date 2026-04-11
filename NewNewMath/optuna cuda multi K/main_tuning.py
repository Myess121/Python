# main_tuning.py
# -*- coding: utf-8 -*-
"""
多无人机数量调参主程序（修复版）
修复：参数传递、评估函数、错误处理
"""

import numpy as np
import random
import time
import json
import optuna
import torch
import logging
from datetime import datetime
from pathlib import Path
from multiprocessing import set_start_method
import traceback
import warnings

warnings.filterwarnings('ignore')

# 导入配置
import config

# 导入模拟器
from uav_simulator import UAVSimulatorCUDA, max_steps, dt, M, N, CUDA_AVAILABLE

# 设置多进程启动方式
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ======================== 评估函数 ========================
def evaluate_params(params, K, N_sim=50, use_parallel=False):
    """评估一组参数（修复版）"""
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # 复制参数避免修改原字典
    params_copy = params.copy()

    # 设置use_pso标志（如果未指定）
    if 'use_pso' not in params_copy:
        params_copy['use_pso'] = True

    # 确保必要参数存在
    if params_copy.get('use_pso', True):
        required_params = ['pop_size', 'max_iter', 'w_start', 'w_end', 'c1', 'c2', 'penalty_coef']
        for p in required_params:
            if p not in params_copy:
                # 使用默认值
                defaults = {
                    'pop_size': 25,
                    'max_iter': 30,
                    'w_start': 0.85,
                    'w_end': 0.25,
                    'c1': 1.5,
                    'c2': 1.5,
                    'penalty_coef': 1000.0
                }
                params_copy[p] = defaults.get(p, 0)

    # 生成初始目标位置（1-based）
    init_positions = [(random.randint(1, M), random.randint(1, N)) for _ in range(N_sim)]
    uav_start = [(1, 1)] * K

    # 创建模拟器
    simulator = UAVSimulatorCUDA()

    # 批量仿真（使用串行模式避免CUDA冲突）
    detect_steps = simulator.batch_simulate(K, init_positions, uav_start, params_copy, n_processes=1)

    detect_steps = np.array(detect_steps)
    avg_step = np.mean(detect_steps)
    avg_time = avg_step * dt

    # 计算各时间段发现率
    time_10h_steps = 10.0 / dt
    time_20h_steps = 20.0 / dt
    time_30h_steps = 30.0 / dt

    rate_5h = np.sum(detect_steps < (5.0 / dt)) / N_sim * 100
    rate_10h = np.sum(detect_steps < time_10h_steps) / N_sim * 100
    rate_15h = np.sum(detect_steps < (15.0 / dt)) / N_sim * 100
    rate_20h = np.sum(detect_steps < time_20h_steps) / N_sim * 100
    rate_30h = np.sum(detect_steps < time_30h_steps) / N_sim * 100
    rate_40h = np.sum(detect_steps < max_steps) / N_sim * 100

    # 清理GPU内存
    del simulator
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()

    return {
        'avg_time_hours': avg_time,
        'detection_rate_5h': rate_5h,
        'detection_rate_10h': rate_10h,
        'detection_rate_15h': rate_15h,
        'detection_rate_20h': rate_20h,
        'detection_rate_30h': rate_30h,
        'detection_rate_40h': rate_40h,
        'all_steps': detect_steps.tolist(),
        'all_times': (detect_steps * dt).tolist()
    }


# ======================== Optuna目标函数 ========================
class Objective:
    def __init__(self, K, N_sim, search_space):
        self.K = K
        self.N_sim = N_sim
        self.search_space = search_space
        self.trial_results = []
        self.best_so_far = float('inf')

    def __call__(self, trial):
        # 从搜索空间采样
        params = {}
        for param_name, param_config in self.search_space.items():
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
        if 'w_end' in params and 'w_start' in params:
            if params['w_end'] >= params['w_start']:
                params['w_end'] = params['w_start'] - 0.05

        # 确保参数在合理范围内
        if params.get('penalty_coef', 1000) < 100:
            params['penalty_coef'] = 100

        logging.info(f"[K={self.K}] Trial {trial.number}: 测试参数 {params}")
        start_time = time.time()

        # 评估参数
        results = evaluate_params(params, self.K, self.N_sim)

        elapsed = time.time() - start_time

        # 记录结果
        trial.set_user_attr('detection_rate_10h', results['detection_rate_10h'])
        trial.set_user_attr('detection_rate_40h', results['detection_rate_40h'])
        trial.set_user_attr('elapsed_seconds', elapsed)

        logging.info(f"[K={self.K}] Trial {trial.number}: "
                     f"时间={results['avg_time_hours']:.2f}h, "
                     f"10h率={results['detection_rate_10h']:.1f}%, "
                     f"耗时={elapsed:.1f}s")

        # 保存结果
        self.trial_results.append({
            'trial_number': trial.number,
            'params': params.copy(),
            'results': results,
            'elapsed': elapsed
        })

        # 更新最佳值
        if results['avg_time_hours'] < self.best_so_far:
            self.best_so_far = results['avg_time_hours']
            logging.info(f"[K={self.K}] 新的最佳值: {self.best_so_far:.2f}h")

        return results['avg_time_hours']


# ======================== 单无人机数量调参 ========================
def tune_for_K(K, n_trials=None, n_sim_per_eval=None):
    """对特定无人机数量进行调参"""

    if n_trials is None:
        n_trials = config.N_TRIALS_PER_K
    if n_sim_per_eval is None:
        n_sim_per_eval = config.N_SIM_PER_EVAL

    print(f"\n{'=' * 60}")
    print(f"开始调参 - 无人机数量: K = {K}")
    print(f"试验次数: {n_trials}, 每次评估仿真次数: {n_sim_per_eval}")
    print(f"{'=' * 60}")

    # 创建结果目录
    result_dir = Path(f"results_K{K}")
    result_dir.mkdir(exist_ok=True)

    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = result_dir / f"tuning_K{K}_{timestamp}.log"

    # 配置日志
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

    # 获取搜索空间
    search_space = config.get_search_space(K)
    logging.info(f"搜索空间: {search_space}")

    # 创建Optuna study
    study_name = f"uav_search_K{K}_{timestamp}"
    storage_url = f"sqlite:///{result_dir}/study_K{K}.db"

    try:
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage_url,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=config.OPTUNA_CONFIG.get('n_startup_trials', 5)
            )
        )
    except Exception as e:
        logging.warning(f"创建study失败，使用内存存储: {e}")
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=config.OPTUNA_CONFIG.get('n_startup_trials', 5)
            )
        )

    # 创建目标函数
    objective = Objective(K, n_sim_per_eval, search_space)

    # 运行优化
    logging.info(f"开始优化 K={K} - 试验次数: {n_trials}")
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logging.info("用户中断优化")
    except Exception as e:
        logging.error(f"优化过程出错: {e}")
        traceback.print_exc()

    # 保存结果
    if study.best_trial:
        save_results(study, objective.trial_results, K, timestamp, result_dir)
        print_best_results(study, K)

        # 返回最佳结果
        return {
            'K': K,
            'timestamp': timestamp,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial_results': get_best_trial_results(study, objective.trial_results),
            'all_trials': objective.trial_results
        }
    else:
        logging.error("没有成功的试验结果")
        return None


def save_results(study, trial_results, K, timestamp, result_dir):
    """保存结果"""
    # 保存最佳参数
    best_config_file = result_dir / f"best_config_K{K}_{timestamp}.json"
    best_config = {
        'K': K,
        'timestamp': timestamp,
        'best_params': study.best_params,
        'best_value': study.best_value,
    }

    # 添加最佳试验的详细结果
    for trial in trial_results:
        if trial['trial_number'] == study.best_trial.number:
            best_config['best_results'] = trial['results']
            break

    with open(best_config_file, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    # 保存完整结果
    results = {
        'K': K,
        'timestamp': timestamp,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'trials': trial_results,
        'search_space': config.get_search_space(K)
    }

    result_file = result_dir / f"results_K{K}_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"结果已保存到: {result_dir}")
    return result_file


def get_best_trial_results(study, trial_results):
    """获取最佳试验的详细结果"""
    for trial in trial_results:
        if trial['trial_number'] == study.best_trial.number:
            return trial['results']
    return None


def print_best_results(study, K):
    """打印最佳结果"""
    logging.info("\n" + "=" * 60)
    logging.info(f"[K={K}] 调参完成！最佳结果")
    logging.info("=" * 60)
    logging.info(f"最佳平均发现时间: {study.best_value:.2f} 小时")
    logging.info("\n最佳参数组合:")
    for param, value in study.best_params.items():
        logging.info(f"  {param} = {value}")


# ======================== 多无人机数量对比 ========================
def run_multi_k_tuning(K_list=None, n_trials_per_k=None, n_sim_per_eval=None):
    """对多个无人机数量进行调参"""

    if K_list is None:
        K_list = config.K_LIST
    if n_trials_per_k is None:
        n_trials_per_k = config.N_TRIALS_PER_K
    if n_sim_per_eval is None:
        n_sim_per_eval = config.N_SIM_PER_EVAL

    all_results = []

    print("\n" + "=" * 80)
    print("多无人机数量调参项目")
    print(f"测试K值: {K_list}")
    print(f"每个K值的试验次数: {n_trials_per_k}")
    print(f"每次评估仿真次数: {n_sim_per_eval}")
    print("=" * 80)

    for idx, K in enumerate(K_list):
        print(f"\n{'#' * 60}")
        print(f"# 正在处理 K = {K} ({idx + 1}/{len(K_list)})")
        print(f"{'#' * 60}")

        result = tune_for_K(K, n_trials=n_trials_per_k, n_sim_per_eval=n_sim_per_eval)
        if result:
            all_results.append(result)

        # 清理GPU内存
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    # 保存汇总结果
    save_summary_results(all_results)

    return all_results


def save_summary_results(all_results):
    """保存所有K值的汇总结果"""
    summary = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': {
            'K_list': config.K_LIST,
            'N_TRIALS_PER_K': config.N_TRIALS_PER_K,
            'N_SIM_PER_EVAL': config.N_SIM_PER_EVAL,
            'RANDOM_SEED': config.RANDOM_SEED
        },
        'results': []
    }

    for result in all_results:
        if result and 'best_trial_results' in result:
            summary['results'].append({
                'K': result['K'],
                'timestamp': result['timestamp'],
                'best_avg_time': result['best_value'],
                'best_params': result['best_params'],
                'detection_rate_10h': result['best_trial_results'].get('detection_rate_10h', 0),
                'detection_rate_20h': result['best_trial_results'].get('detection_rate_20h', 0),
                'detection_rate_40h': result['best_trial_results'].get('detection_rate_40h', 0)
            })

    # 按K值排序
    summary['results'].sort(key=lambda x: x['K'])

    # 保存到文件
    summary_file = Path("summary_all_K.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("汇总结果 - 不同无人机数量对比")
    print("=" * 80)
    print(f"{'K':<4} {'最佳平均时间(h)':<15} {'10h发现率(%)':<12} {'20h发现率(%)':<12} {'40h发现率(%)':<12}")
    print("-" * 80)
    for r in summary['results']:
        print(f"{r['K']:<4} {r['best_avg_time']:<15.2f} {r['detection_rate_10h']:<12.1f} "
              f"{r['detection_rate_20h']:<12.1f} {r['detection_rate_40h']:<12.1f}")
    print("=" * 80)

    print(f"\n汇总结果已保存到: {summary_file}")


# ======================== 快速测试单个K值 ========================
def quick_test_K(K=7, n_trials=5, n_sim_per_eval=20):
    """快速测试单个K值（用于调试）"""
    return tune_for_K(K, n_trials=n_trials, n_sim_per_eval=n_sim_per_eval)


# ======================== 主函数 ========================
def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("多无人机数量调参程序（修复版）")
    print(f"CUDA可用: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # 显示当前配置
    config.print_config()

    # 询问运行模式
    print("\n请选择运行模式:")
    print("1. 完整运行 (使用config.py中的所有K值)")
    print("2. 快速测试 (单个K值，5次试验) - 用于调试")
    print("3. 自定义运行 (手动指定K值和试验次数)")

    choice = input("\n请输入选择 (1/2/3): ").strip()

    if choice == '1':
        run_multi_k_tuning()

    elif choice == '2':
        test_K = int(input(f"请输入测试的K值 (2-9，默认7): ").strip() or "7")
        quick_test_K(K=test_K, n_trials=5, n_sim_per_eval=20)

    elif choice == '3':
        k_input = input(f"请输入K值列表，用逗号分隔 (默认{config.K_LIST}): ").strip()
        if k_input:
            K_list = [int(x.strip()) for x in k_input.split(',')]
        else:
            K_list = config.K_LIST

        n_trials = int(
            input(f"请输入每个K值的试验次数 (默认{config.N_TRIALS_PER_K}): ").strip() or str(config.N_TRIALS_PER_K))
        n_sim = int(
            input(f"请输入每次评估的仿真次数 (默认{config.N_SIM_PER_EVAL}): ").strip() or str(config.N_SIM_PER_EVAL))

        run_multi_k_tuning(
            K_list=K_list,
            n_trials_per_k=n_trials,
            n_sim_per_eval=n_sim
        )

    else:
        print("无效选择，使用默认配置运行快速测试...")
        quick_test_K(K=7, n_trials=5, n_sim_per_eval=20)

    print("\n程序执行完成！")


if __name__ == "__main__":
    main()