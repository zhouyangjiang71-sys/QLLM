import ast
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
from LLM_helper import*
from run import REGISTRY as run_REGISTRY
from envs import REGISTRY as env_REGISTRY
from types import SimpleNamespace as SN
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()
logger.setLevel("INFO")
ex = Experiment("pymarl", save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    run_REGISTRY[_config['run']](_run, config, _log)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config

if __name__ == '__main__':
    params = deepcopy(sys.argv)

    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    LLM_args=LLM_args_manage()

    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")

    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
            config_dict["env_args"]["map_name"] = map_name
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]
            config_dict["env_args"]["key"] = map_name

    args = SN(**config_dict)
    env = env_REGISTRY[args.env](
        **args.env_args,
        common_reward=args.common_reward,
        reward_scalarisation=args.reward_scalarisation,
    )
    if ':penalty' in map_name:
        map_name = map_name.replace(":penalty", "")# Windows computers can't download files with illegal characters':'
        LLM_args.map = 'matrix_game'
    elif "lbforaging" in map_name:
        map_name = map_name.replace("lbforaging:Foraging-", "")
        LLM_args.map = map_name
    elif "mpe" in map_name:
        map_name = map_name.replace("pz-mpe-", "")
        LLM_args.map = map_name
    LLM_args.n_agents = env.n_agents
    LLM_args.state_shape = env.get_state_size()
    print(f"env.n_agents:{env.n_agents}, env.state_shape:{env.get_state_size()}")

    with open('src/qllm.txt', 'r', encoding='utf-8') as file:
        qllm_setup = file.read()
    with open('src/modules/mixers/qllm.py', 'w') as file:
        file.write(qllm_setup)
    with open('src/setup_prompt_'+LLM_args.map+'.txt', 'r', encoding='utf-8') as file:
        setup_prompt = file.read()
    with open('src/system_prompt.txt', 'r', encoding='utf-8') as file:
        system_prompt = file.read()

    LLM_args.maker_num=config_dict["maker_num"]
    LLM_args.message_length=config_dict["message_length"]
    My_maker = QLLM_maker(LLM_args,system_prompt+setup_prompt)
    My_evaluator = LLM_evaluator(setup_prompt,LLM_args.map)
    if config_dict["LLM_pretrain"]:
        QLLM_list = []
        for t in range(config_dict["LLM_episode"]):
            print(f"TFCAF update for round {t + 1}")
            while len(My_maker.messages) > LLM_args.message_length:
                My_maker.messages.pop(1)
                My_maker.messages.pop(1)
            while len(My_evaluator.messages) > LLM_args.message_length:
                My_evaluator.messages.pop(1)
                My_evaluator.messages.pop(1)
            QLLM_list = []
            if t == 0:
                advice = ''
            else:
                advice = My_evaluator.maker(
                    "Please read carefully the meaning of each component of the global state space in the description of the multi-agent reinforcement learning task and read the implementation details of your chosen function carefully.\
                    Then, suggest one modification based on the function you have chosen. Your advice is invaluable to me, and the following requirements need to be adhered to:\
                    (1)Your advice should be one sentence only, not too long.\
                    (2)Your suggestion needs to considering all possible scenarios in a battle and developing strategies for them, making the calculation of weights more precise.\
                    (3)You need to carefully read the function you have chosen and analyze the potential errors that may occur in it.")
            for i in range(LLM_args.maker_num):
                if t!=0:
                    temp = My_maker.maker(
                        "Here are some tips for modifying functions:(1)If the magnitude of a weight is significantly larger, the value must be remapped to the appropriate range"\
                        + f"(2)This tip may or may not be useful, so please analyse it carefully before deciding whether or not to implement it:\n{advice}\
                        (3)You can modify the formula for some weights, improve some hyperparameters of the function, or add component of calculating weights.\
                        (4)The code you output must start with ``` python and end with ```.Do not include any text or explanations.\
                        Then, give an improved credit assignment function after carefully reading the task description and highlights again.\
                        Be sure to double-check for dimension matching issues in function calculations as well as system-required function input and output specifications.")
                    QLLM_list.append(temp)
                else:
                    temp = My_maker.maker(
                        "Here are some tips for writing credit assignment functions:\
                        (1)If the magnitude of a weight is significantly larger, the value must be remapped to the appropriate range.\
                        (2)The code you output must start with ``` python and end with ```.Do not include any text or explanations.\
                        Please write the complete QLLMNetwork credit assignment function after carefully reading the task description and key scenarios.\
                        Be sure to double-check for dimension matching issues in function calculations as well as system-required function input and output specifications.")
                    QLLM_list.append(temp)
            QLLM_choose = My_evaluator.maker(
                f"I have designed {LLM_args.maker_num} candidate QLLM functions as follows:{QLLM_list}.\
                Could you please output only an array of length {LLM_args.maker_num} containing only 0 and 1. \
                0 means you don't select the function, 1 means you select the function.You can only select one function")
            QLLM_choose_list = ast.literal_eval(QLLM_choose)
            if 1 in QLLM_choose_list:
                My_maker.addmemory(QLLM_list[QLLM_choose_list.index(1)])
            else:
                My_maker.addmemory(QLLM_list[0])
        with open('src/' + 'TFCAF_' +LLM_args.map + '.txt', "w", encoding="utf-8") as f:
            f.write(My_maker.messages[-1]["content"])
        QLLM=My_maker.messages[-1]["content"]
    else:
        with open('src/' + 'TFCAF_' +LLM_args.map + '.txt', 'r') as f:
            QLLM = f.read()
    loadQLLM(QLLM)
    ex.add_config(config_dict)
    file_obs_path = os.path.join(results_path, f"sacred/{config_dict['name']}/{map_name}")
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)


