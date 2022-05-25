import numpy as np
import os
import gym
import ma_gym
from multiagent import MultiAgent


if __name__ == '__main__':

	seed_num = 0
	extension = "MAPPO_Q" # [MAPPO_Q, MAA2C_Q, MAPPO_Q_Semi_Hard_Attn, MAA2C_Q_Semi_Hard_Attn]
	test_num = "traffic_junction"
	env_name = "ma_gym:TrafficJunction10-v0"
	experiment_type = "prd_above_threshold" # shared, prd_above_threshold

	dictionary = {
			"iteration": seed_num,
			"update_type": "ppo", # [a2c, ppo]
			"grad_clip_critic": 0.5,
			"grad_clip_actor": 0.5,
			"attention_type": 'soft', # [soft, semi-hard]
			"device": "gpu",
			"critic_dir": '../../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
			"actor_dir": '../../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
			"gif_dir": '../../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_eval_dir":'../../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_clip": 0.05,
			"value_clip": 0.05,
			"n_epochs": 5,
			"update_ppo_agent": 2, # update ppo agent after every 'update_ppo_agent' episodes
			"env": env_name, 
			"test_num":test_num,
			"extension":extension,
			"value_lr": 5e-5,
			"policy_lr": 5e-5,
			"entropy_pen": 0.0,
			"gamma": 0.99, 
			"gae_lambda": 0.95,
			"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
			"select_above_threshold": 0.1, #[soft: 0.1, semi-hard: 0.0 --> attention_type]
			"gif": False,
			"gif_checkpoint":1,
			"load_models": False,
			"model_path_value": "",
			"model_path_policy": "",
			"eval_policy": False,
			"save_model": False,
			"save_model_checkpoint": 10,
			"save_comet_ml_plot": False,
			"learn":True,
			"max_episodes": 20000,
			"max_time_steps": 40,
			"experiment_type": experiment_type,
		}
	env = gym.make(env_name)
	ma_controller = MAPPO(env,dictionary)
	ma_controller.run()