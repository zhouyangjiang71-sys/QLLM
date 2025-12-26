import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
# from smac.env import StarCraft2Env
import warnings
# def get_agent_own_state_size(env_args):
#     sc_env = StarCraft2Env(**env_args)
#     print(sc_env.map_name)
#     # qatten parameter setting (only use in qatten)
#     return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits
#训练开始
def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)               # 对配置参数进行合理性检查。通过args_sanity_check函数验证_config的各项参数是否设置正确，以避免运行时发生错误
    args = SN(**_config)                                     # 将_config转换为SimpleNamespace对象，允许通过点操作符访问配置参数。
    args.device = "cuda" if args.use_cuda else "cpu"         # 根据配置中的use_cuda参数，决定是否使用GPU（cuda）还是CPU

    logger = Logger(_log)                                    # 创建一个Logger对象
    _log.info("Experiment Parameters:")                      # 记录实验的配置信息
    experiment_params = pprint.pformat(_config, indent=4, width=1) # 美化配置信息，使其在日志输出中更加易读
    _log.info("\n\n" + experiment_params + "\n")             # 将配置信息记录到日志中

    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) # 生成一个唯一的实验标识符unique_token，由实验名称和当前时间组成
    args.unique_token = unique_token                         # 将其保存到args中，用于标识实验
    if args.use_tensorboard:                                 # 如果配置中开启了Tensorboard(args.use_tensorboard)，则
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs") # 定义Tensorboard日志的存储路径tb_logs_direc和具体实验目录tb_exp_direc
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)                                                           # 使用logger.setup_tb(tb_exp_direc)初始化Tensorboard日志记录器

    logger.setup_sacred(_run)                                # 配置Sacred框架，使用logger.setup_sacred(_run)，让Sacred记录日志等实验信息

    run_sequential(args=args, logger=logger)                 # 调用run_sequential函数，传入配置参数args和日志记录器logger


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    #parallel_runner or episode_runner
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    # if getattr(args, 'agent_own_state_size', False):
    #     args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # 定义scheme，用于描述不同数据的形状（例如状态、观测、动作、奖励等）和数据类型。groups用于定义智能体组（即多智能体的数量），preprocess负责对动作进行预处理（例如将动作进行one-hot编码）
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # 初始化多智能体控制器mac，从注册表mac_REGISTRY中根据args.mac加载控制器。控制器负责为各个智能体选择动作，并根据策略进行决策
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    # 初始化一些变量以控制训练过程，包括当前episode编号、上次测试和日志的时间步数、模型保存的时间步数等，并记录训练开始时间
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue
            episode_sample = None
            while True:
                episode_sample = buffer.sample(args.batch_size)
                if len(episode_sample["state"])==0:
                    time.sleep(1)  # 等待缓冲区填充
                    warnings.warn("Buffer sampling is empty!", UserWarning)
                    continue
                else:
                    break

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")
    print("Exiting Main")  # 训练结束后的清理操作。首先打印一条信息表示主程序即将退出。
    print("Stopping all threads")  # 遍历所有正在运行的线程，确保除主线程外的其他线程都安全地关闭。对于每个非主线程，使用join方法等待其结束，避免程序强制退出时导致线程不安全退出
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")
    print("Exiting script")

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
