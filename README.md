# Installing Dependencies

To install the dependencies for the codebase, clone this repo and run:
```sh
pip install -r requirements.txt
```

To install a set of supported environments, you can run:
```sh
cd lb-foraging-master
pip install -e .
```
```sh
pip install pettingzoo
```
```sh
cd matrix-games-master
pip install -e .
```
```sh
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
pip install --upgrade psutil wheel pytest
pip install gfootball==2.10.2 gym==0.11
```

which will install the following environments:
- [Level Based Foraging](https://github.com/uoe-agents/lb-foraging)
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) (used for the multi-agent particle environment)
- [Matrix games](https://github.com/uoe-agents/matrix-games)
- [Google Research Football](https://github.com/google-research/football)

# Run instructions
To run the baseline algorithm experiments, you can use the following command:

Matrix games:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="matrixgames:penalty-100-nostate-v0"
```

LBF:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```

MPE:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3"
```

Google Research Football(**run in G-football file**):
```sh
python src/main.py --config=qmix --env-config=gfootball with env_args.time_limit=150 env_args.map_name="academy_counterattack_easy"
```
Note that for the MPE environments tag (predator-prey) and adversary, we provide pre-trained prey and adversary policies. These can be used to control the respective agents to make these tasks fully cooperative (used in the paper) by setting `env_args.pretrained_wrapper="PretrainedTag"` or `env_args.pretrained_wrapper="PretrainedAdversary"`.

Below, we provide the base environment and key / map name for all the environments evaluated in our paper:

- Matrix games: all with `--env-config=gymma with env_args.time_limit=25 env_args.key="..."`
  - Climbing: `matrixgames:climbing-nostate-v0`
- LBF: all with `--env-config=gymma with env_args.time_limit=50 env_args.key="..."`
  - 8x8-2p-2f-2s-coop: `lbforaging:Foraging-2s-8x8-2p-2f-coop-v3`
  - 10x10-3p-3f-2s: `lbforaging:Foraging-2s-10x10-3p-3f-v3`
  - 15x15-4p-3f-2s: `lbforaging:Foraging-15x15-4p-3f-v3`
- MPE: all with `--env-config=gymma with env_args.time_limit=25 env_args.key="..."`
  - simple spread: `pz-mpe-simple-spread-v3`
  - simple adversary: `pz-mpe-simple-adversary-v3` with additional `env_args.pretrained_wrapper="PretrainedAdversary"`
  - simple tag: `pz-mpe-simple-tag-v3` with additional `env_args.pretrained_wrapper="PretrainedTag"`
- G-football: all with `--env-config=gfootball with env_args.time_limit=150 env_args.map_name="..."`
  - `academy_pass_and_shoot_with_keeper`
  - `academy_3_vs_1_with_keeper`
  - `academy_counterattack_easy`

To run the QLLM algorithm, the main function is changed to qllm_main.py and the running instructions are as follows:
```sh
python src/qllm_main.py --config=qllm --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3"
```
After executing the above commands, a TFCAF text file corresponding to the environment will be generated in the `src` folder. This file can be used for future runs.

In the QLLM configuration file `qllm.yaml`, the following parameters are defined:

- **`LLM_pretrain`**: Determines whether the LLM should regenerate the TFCAF or directly import an existing one.
- **`LLM_episode`**: Specifies the number of training iterations.
- **`maker_num`**: Indicates the number of candidate TFCAFs generated in each iteration.
- **`message_length`**: Defines the maximum memory capacity of the LLM.

**Before running QLLM, you need to fill in the corresponding api key,base_url and model name of deepseek or chatgpt into the corresponding location in `LLM_helper.py`.
You can log on to (https://platform.deepseek.com/) and (https://platform.openai.com/) to get the api key.**

