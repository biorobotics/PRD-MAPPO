<!-- **Status:** Archive (code is provided as-is, no updates expected) -->

# Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization

This is the code for implementing the PRD-MAPPO algorithm presented in the paper:
[Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization]().
It is configured to be run in conjunction with the following environments:
- [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).
- [Pressure Plate (PP)](https://github.com/uoe-agents/pressureplate)
- [MultiAgent Gym (MA-GYM)](https://github.com/koulanurag/ma-gym)
Note: Please note that the environments listed above are customised and hence one should use the environment directories provided in the above codebase


## Installation

- To install MPE, PP, or MA-GYM, `cd` into the root directory and type `pip install -e .`

- Known dependencies for MPE: Python (3.6+), OpenAI gym (0.10.5), torch (1.10.0+cu102), numpy (1.21.5)

- Known dependencies for PP: Python (3.6+), OpenAI gym (0.23.1), torch (1.11.0+cu102), numpy (1.22.3)

- Known dependencies for MA-GYM: Python (3.6+), OpenAI gym (0.19.0), torch (1.11.0+cu102), numpy (1.22.3)

## Core training parameters

You can find these parameters in the `main.py` file for all the environments.

- `iteration`: seed index (default: `0`, options: `0, 1, 2, 3, 4`)

- `update_type`: policy update algorithm (default: `ppo`, options: `ppo, a2c`)

- `attention_type`: transformer attention mechanism for the critic network (default: `soft`, options: `soft, semi-hard`)

- `device`: device to run the code on (default: `gpu`, option: `gpu, cpu`)

- `grad_clip_critic`: gradient clip for critic network (default: `10.0 (MPE) or 0.5 (MA-GYM/PP)`)

- `grad_clip_actor`: gradient clip for actor network (default: `10.0 (MPE) or 0.5 (MA-GYM/PP)`)

- `critic_dir`: directory to save critic network models

- `actor_dir`: directory to save actor network models

- `gif_dir`: directory to save gifs

- `policy_eval_dir`: directory to save policy metrics

- `policy_clip`: imposes a clip interval on the probability ratio term while computing policy loss, which is clipped into a range [1 — policy_clip, 1 + policy_clip] (default: `0.05`)

- `value_clip`: imposes a clip interval on the probability ratio term while computing value loss, which is clipped into a range [1 — value_clip, 1 + value_clip] (default: `0.05`)

- `n_epochs`: number of epochs to train the policy and critic network (default: `5`)

- `env`: environment name

- `value_lr`: critic learning rate (default: `1e-3 (Crossing) or 3e-4 (Combat) or 7e-4 (Pressure Plate) or 5e-5 (Traffic Junction)`)

- `policy_lr`: actor learning rate (default: `7e-4 (Crossing) or 3e-4 (Combat) or 7e-4 (Pressure Plate) or 5e-5 (Traffic Junction)`)

- `entropy_pen`: entropy penalty (default: `0.0 (Crossing) or 8e-3 (Combat) or 0.4 (Pressure Plate) or 0.0 (Traffic Junction)`)

- `gamma`: discount factor (default: `0.99`)

- `gae_lambda`: temperature factor for Generalized Advantage Estimation (default: `0.95`)

- `lambda`: temperature factor for computing TD-lambda targets (default: `0.95`)

- `select_above_threshold`: weight threshold to identify relevant set (default: `0.05 (Crossing) or 0.2 (Combat) or 0.05 (Pressure Plate) or 0.2 (Traffic Junction)`)

- `gif`: enable rendering of gif

- `gif_checkpoint`: episodes after which render gif (default: `1`)

- `load_models`: enable to load critic and actor models

- `model_path_value`: critic model path

- `model_path_policy`: actor model path

- `eval_policy`: enable to capture policy evaluation metrics

- `save_model`: enable to save critic and actor models

- `save_model_checkpoint`: save model after `save_model_checkpoint` episodes

- `save_comet_ml_plot`: enable to record data on comet

- `learn`: enable updating critic and actor networks

- `max_episodes`: total number of episodes (default: `80K (Crossing) or 120K (Combat) or 20K (Pressure Plate) or 20K (Traffic Junction)`)

- `max_time_steps`: number of timesteps per episode (default: `50 (Crossing) or 40 (Combat) or 70 (Pressure Plate) or 40 (Traffic Junction)`)

- `experiment_type`: type of update (default: `prd`, options: `prd, shared (fully cooperative)`)


## Code structure

- `./Agent MA GYM/MA_Controller/Combat/main.py`: contains code for setting parameters of PRD-MAPPO on the MA-GYM Combat environment

- `./Agent MA GYM/MA_Controller/Traffic_Junc/main.py`: contains code for setting parameters of PRD-MAPPO on the MA-GYM Traffic Junction environment

- `./Agent MPE/MA_Controller/main.py`: contains code for setting parameters of PRD-MAPPO on the MPE Crossing environment

- `./Agent Pressure Plate/MA_Controller/main.py`: contains code for setting parameters of PRD-MAPPO on the PP 4 Person Pressure Plate environment

- `./Agent MA GYM/MA_Controller/Combat/agent.py` or `./Agent MA GYM/MA_Controller/Traffic_Junc/agent.py` or `./Agent Pressure Plate/MA_Controller/agent.py` or `./Agent MPE/MA_Controller/agent.py`: core code for the PRD-MAPPO algorithm

- `./Agent MA GYM/MA_Controller/Combat/multiagent.py` or `./Agent MA GYM/MA_Controller/Traffic_Junc/multiagent.py` or `./Agent Pressure Plate/MA_Controller/multiagent.py` or `./Agent MPE/MA_Controller/multiagent.py`: code that deals with environment and agent interaction

- `./Agent MA GYM/MA_Controller/Combat/model.py` or `./Agent MA GYM/MA_Controller/Traffic_Junc/model.py` or `./Agent Pressure Plate/MA_Controller/model.py` or `./Agent MPE/MA_Controller/model.py`: Policy, Q Network, Replay Buffer code for PRD-MAPPO
