import numpy as np
import os
import gym
import ma_gym
from multiagent import MultiAgent


if __name__ == '__main__':
	seed_num = 0 # [0,1,2,3,4]
	extension = "MAPPO_Q" # [MAPPO_Q, MAA2C_Q, MAPPO_Q_Semi_Hard_Attn, MAA2C_Q_Semi_Hard_Attn]
	test_num = "combat"
	env_name = "ma_gym:Combat-v0"
	experiment_type = "prd_above_threshold" # shared, prd_above_threshold

	dictionary = {
			"iteration": seed_num,
			"update_type": "ppo", #[ppo, a2c]
			"attention_type": 'soft', #[soft, semi-hard]
			"grad_clip_critic": 0.5,
			"grad_clip_actor": 0.5,
			"device": "gpu",
			"critic_dir": '../../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
			"actor_dir": '../../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
			"gif_dir": '../../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_eval_dir":'../../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_clip": 0.05,
			"value_clip": 0.05,
			"n_epochs": 5,
			"update_ppo_agent": 1, # update ppo agent after every update_ppo_agent episodes
			"env": env_name, 
			"test_num":test_num,
			"extension":extension,
			"value_lr": 3e-4, #1e-3
			"policy_lr": 3e-4, #prd 1e-4
			"entropy_pen": 8e-3, #8e-3
			"gamma": 0.99, 
			"gae_lambda": 0.95,
			"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
			"select_above_threshold": 0.2, # [when attention_type- "soft": 0.2, "semi-hard": 0.0]
			"gif": False,
			"gif_checkpoint":1,
			"load_models": False,
			"model_path_value": "",
			"model_path_policy": "",
			"eval_policy": False,
			"save_model": False,
			"save_model_checkpoint": 1000,
			"save_comet_ml_plot": False,
			"learn":True,
			"max_episodes": 120000,
			"max_time_steps": 40,
			"experiment_type": experiment_type,
		}
	env = gym.make(env_name)
	ma_controller = MultiAgent(env,dictionary)
	ma_controller.run()