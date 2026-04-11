# run_tuning.py
# -*- coding: utf-8 -*-
"""
CC-MPSO参数调优主程序
使用Optuna自动搜索最优CC-MPSO参数
"""

import numpy as np
import random
import time
import json
import optuna
import logging
import traceback
from datetime import datetime
from pathlib import Path
from multiprocessing import set_start_method, cpu_count, Pool
import warnings

warnings.filterwarnings('ignore')

# 导入配置
import config
from config import (
    M, N, dt, max_steps, start_step,
    Pd, Pf, p_stay, p_move,
    K_LIST, N_TRIALS_PER_K, N_SIM_PER_EVAL, RANDOM_SEED,
    DEFAULT_CCMPSO_PARAMS, get_search_space
)

# 导入模拟器
from uav_simulator import UAVSimulator, _run_simulation_worker

# 设置多进程启动方式
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ======================== 评估函数 ========================
def evaluate_params(params, K, N_sim=N_SIM_PER_EVAL, use_parallel=False, n_processes=None):
    """
    评估一组CC-MPSO参数

    参数:
        params: 参数字典
        K: 无人机数量
        N_sim: 仿真次数
        use_parallel: 是否使用并行
        n_processes: 并行进程数

    返回:
        评估结果字典
    """
    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 生成初始目标位置（1-based）
    init_positions = [(random.randint(1, M), random.randint(1, N)) for _ in range(N_sim)]
    uav_start = [(1, 1)] * K

    # 合并参数（使用提供的参数，缺失的使用默认值）
    sim_params = DEFAULT_CCMPSO_PARAMS.copy()
    sim_params.update(params)
    sim_params['use_pso'] = True  # 确保使用CC-MPSO

    if use_parallel and n_processes and n_processes > 1:
        # 并行执行
        args_list = [(K, init_pos, uav_start, sim_params) for init_pos in init_positions]
        with Pool(processes=n_processes) as pool:
            detect_steps = list(pool.map(_run_simulation_worker, args_list))
    else:
        # 串行执行
        simulator = UAVSimulator()
        detect_steps = simulator.batch_simulate(K, init_positions, uav_start, sim_params)

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

    return {
        'avg_time_hours': avg_time,
        'avg_steps': float(avg_step),
        'detection_rate_5h': rate_5h,
        'detection_rate_10h': rate_10h,
        'detection_rate_15h': rate_15h,
        'detection_rate_20h': rate_20h,
        'detection_rate_30h': rate_30h,
        'detection_rate_40h': rate_40h,
        'all_steps': detect_steps.tolist()
    }


# ======================== Optuna目标函数 ========================
class CCMPSOObjective:
    """CC-MPSO参数优化的Optuna目标函数"""

    def __init__(self, K, N_sim, search_space, use_parallel=False, n_processes=None):
        self.K = K
        self.N_sim = N_sim
        self.search_space = search_space
        self.use_parallel = use_parallel
        self.n_processes = n_processes
        self.trial_results = []
        self.best_so_far = float('inf')

    def __call__(self, trial):
        # 从搜索空间采样参数
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
        params['penalty_coef'] = max(params.get('penalty_coef', 500), 100)

        logging.info(f"[K={self.K}] Trial {trial.number}: 测试参数 {params}")
        start_time = time.time()

        # 评估参数
        try:
            results = evaluate_params(
                params, self.K, self.N_sim,
                use_parallel=self.use_parallel,
                n_processes=self.n_processes
            )
        except Exception as e:
            logging.error(f"评估失败: {e}")
            traceback.print_exc()
            return float('inf')

        elapsed = time.time() - start_time

        # 记录结果
        trial.set_user_attr('detection_rate_10h', results['detection_rate_10h'])
        trial.set_user_attr('detection_rate_20h', results['detection_rate_20h'])
        trial.set_user_attr('detection_rate_40h', results['detection_rate_40h'])
        trial.set_user_attr('elapsed_seconds', elapsed)
        trial.set_user_attr('params', params)

        logging.info(f"[K={self.K}] Trial {trial.number}: "
                     f"时间={results['avg_time_hours']:.2f}h, "
                     f"10h率={results['detection_rate_10h']:.1f}%, "
                     f"40h率={results['detection_rate_40h']:.1f}%, "
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


# ======================== 单K值调参 ========================
def tune_for_K(K, n_trials=None, n_sim_per_eval=None, use_parallel=False, n_processes=None):
    """
    对特定无人机数量进行CC-MPSO参数调优

    参数:
        K: 无人机数量
        n_trials: Optuna试验次数
        n_sim_per_eval: 每次评估的仿真次数
        use_parallel: 是否使用并行
        n_processes: 并行进程数

    返回:
        调优结果字典
    """
    if n_trials is None:
        n_trials = N_TRIALS_PER_K
    if n_sim_per_eval is None:
        n_sim_per_eval = N_SIM_PER_EVAL
    if n_processes is None:
        n_processes = min(cpu_count(), 4)

    print(f"\n{'=' * 60}")
    print(f"开始CC-MPSO参数调优 - 无人机数量: K = {K}")
    print(f"试验次数: {n_trials}, 每次评估仿真次数: {n_sim_per_eval}")
    print(f"并行模式: {use_parallel}, 进程数: {n_processes if use_parallel else 1}")
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
    search_space = get_search_space(K)
    logging.info(f"搜索空间: {search_space}")

    # 创建Optuna study
    study_name = f"ccmpson_K{K}_{timestamp}"
    storage_url = f"sqlite:///{result_dir}/study_K{K}.db"

    try:
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage_url,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=config.OPTUNA_CONFIG.get('n_startup_trials', 5)
            )
        )
    except Exception as e:
        logging.warning(f"创建study失败，使用内存存储: {e}")
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=config.OPTUNA_CONFIG.get('n_startup_trials', 5)
            )
        )

    # 创建目标函数
    objective = CCMPSOObjective(
        K, n_sim_per_eval, search_space,
        use_parallel=use_parallel,
        n_processes=n_processes
    )

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
    """保存调优结果"""
    # 保存最佳参数
    best_config_file = result_dir / f"best_config_K{K}_{timestamp}.json"
    best_config = {
        'K': K,
        'timestamp': timestamp,
        'algorithm': 'CC-MPSO',
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
        'algorithm': 'CC-MPSO',
        'best_params': study.best_params,
        'best_value': study.best_value,
        'trials': trial_results,
        'search_space': get_search_space(K)
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
    logging.info(f"[K={K}] CC-MPSO调参完成！最佳结果")
    logging.info("=" * 60)
    logging.info(f"最佳平均发现时间: {study.best_value:.2f} 小时")
    logging.info("\n最佳参数组合:")
    for param, value in study.best_params.items():
        logging.info(f"  {param} = {value}")


# ======================== 多K值批量调优 ========================
def run_multi_k_tuning(K_list=None, n_trials_per_k=None, n_sim_per_eval=None,
                       use_parallel=False, n_processes=None):
    """
    对多个无人机数量进行CC-MPSO参数调优
    """
    if K_list is None:
        K_list = config.K_LIST
    if n_trials_per_k is None:
        n_trials_per_k = config.N_TRIALS_PER_K
    if n_sim_per_eval is None:
        n_sim_per_eval = config.N_SIM_PER_EVAL
    if n_processes is None:
        n_processes = min(cpu_count(), 4)

    all_results = []

    print("\n" + "=" * 80)
    print("CC-MPSO 多无人机数量参数调优项目")
    print(f"测试K值: {K_list}")
    print(f"每个K值的试验次数: {n_trials_per_k}")
    print(f"每次评估仿真次数: {n_sim_per_eval}")
    print(f"并行模式: {use_parallel}")
    print("=" * 80)

    for idx, K in enumerate(K_list):
        print(f"\n{'#' * 60}")
        print(f"# 正在处理 K = {K} ({idx + 1}/{len(K_list)})")
        print(f"{'#' * 60}")

        result = tune_for_K(
            K,
            n_trials=n_trials_per_k,
            n_sim_per_eval=n_sim_per_eval,
            use_parallel=use_parallel,
            n_processes=n_processes
        )
        if result:
            all_results.append(result)

    # 保存汇总结果
    save_summary_results(all_results)

    return all_results


def save_summary_results(all_results):
    """保存所有K值的汇总结果"""
    summary = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'algorithm': 'CC-MPSO',
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
    summary_file = Path("summary_ccmpson_tuning.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("CC-MPSO调优汇总结果 - 不同无人机数量对比")
    print("=" * 80)
    print(f"{'K':<4} {'最佳平均时间(h)':<15} {'10h发现率(%)':<12} {'20h发现率(%)':<12} {'40h发现率(%)':<12}")
    print("-" * 80)
    for r in summary['results']:
        print(f"{r['K']:<4} {r['best_avg_time']:<15.2f} {r['detection_rate_10h']:<12.1f} "
              f"{r['detection_rate_20h']:<12.1f} {r['detection_rate_40h']:<12.1f}")
    print("=" * 80)

    print(f"\n汇总结果已保存到: {summary_file}")


# ======================== 快速测试 ========================
def quick_test(K=7, n_trials=5, n_sim_per_eval=20, use_parallel=False):
    """快速测试单个K值（用于调试）"""
    return tune_for_K(
        K,
        n_trials=n_trials,
        n_sim_per_eval=n_sim_per_eval,
        use_parallel=use_parallel
    )


def benchmark_default_params(K=7, N_sim=100):
    """测试默认参数的性能（作为基线）"""
    print(f"\n测试默认参数: K={K}, N_sim={N_sim}")

    from uav_simulator import UAVSimulator
    simulator = UAVSimulator()

    random.seed(RANDOM_SEED)
    init_positions = [(random.randint(1, M), random.randint(1, N)) for _ in range(N_sim)]
    uav_start = [(1, 1)] * K

    detect_steps = simulator.batch_simulate(K, init_positions, uav_start, verbose=True)

    avg_time = np.mean(detect_steps) * dt
    rate_10h = np.sum(detect_steps < (10.0 / dt)) / N_sim * 100
    rate_40h = np.sum(detect_steps < max_steps) / N_sim * 100

    print(f"\n默认参数结果:")
    print(f"  平均发现时间: {avg_time:.2f} 小时")
    print(f"  10小时发现率: {rate_10h:.1f}%")
    print(f"  40小时发现率: {rate_40h:.1f}%")

    return {
        'avg_time': avg_time,
        'rate_10h': rate_10h,
        'rate_40h': rate_40h
    }


# ======================== 主函数 ========================
def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("CC-MPSO 参数调优程序")
    print("协同进化多粒子群优化 (Cooperative Co-evolutionary MPSO)")
    print("=" * 60)

    # 显示当前配置
    config.print_config()

    # 询问运行模式
    print("\n请选择运行模式:")
    print("1. 完整运行 (使用config.py中的所有K值)")
    print("2. 快速测试 (单个K值，5次试验) - 用于调试")
    print("3. 自定义运行 (手动指定K值和试验次数)")
    print("4. 基准测试 (测试默认参数性能)")

    choice = input("\n请输入选择 (1/2/3/4): ").strip()

    if choice == '1':
        # 询问是否使用并行
        use_parallel = input("是否使用并行计算? (y/n, 默认n): ").strip().lower() == 'y'
        run_multi_k_tuning(use_parallel=use_parallel)

    elif choice == '2':
        test_K = int(input(f"请输入测试的K值 (2-9，默认7): ").strip() or "7")
        use_parallel = input("是否使用并行计算? (y/n, 默认n): ").strip().lower() == 'y'
        quick_test(K=test_K, n_trials=5, n_sim_per_eval=20, use_parallel=use_parallel)

    elif choice == '3':
        k_input = input(f"请输入K值列表，用逗号分隔 (默认{config.K_LIST}): ").strip()
        if k_input:
            K_list = [int(x.strip()) for x in k_input.split(',')]
        else:
            K_list = config.K_LIST

        n_trials = int(
            input(f"请输入每个K值的试验次数 (默认{config.N_TRIALS_PER_K}): ").strip()
            or str(config.N_TRIALS_PER_K)
        )
        n_sim = int(
            input(f"请输入每次评估的仿真次数 (默认{config.N_SIM_PER_EVAL}): ").strip()
            or str(config.N_SIM_PER_EVAL)
        )
        use_parallel = input("是否使用并行计算? (y/n, 默认n): ").strip().lower() == 'y'

        run_multi_k_tuning(
            K_list=K_list,
            n_trials_per_k=n_trials,
            n_sim_per_eval=n_sim,
            use_parallel=use_parallel
        )

    elif choice == '4':
        test_K = int(input(f"请输入测试的K值 (2-9，默认7): ").strip() or "7")
        N_sim = int(input(f"请输入仿真次数 (默认100): ").strip() or "100")
        benchmark_default_params(K=test_K, N_sim=N_sim)

    else:
        print("无效选择，使用默认配置运行快速测试...")
        quick_test(K=7, n_trials=5, n_sim_per_eval=20)

    print("\n程序执行完成！")


if __name__ == "__main__":
    main()