from multiagent import MultiAgent
import os
import numpy as np
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env


if __name__ == '__main__':
	seed_num = 0 # [0,1,2,3,4]

	extension = "MAPPO_Q" # [MAPPO_Q, MAA2C_Q, MAPPO_Q_Semi_Hard_Attn, MAA2C_Q_Semi_Hard_Attn]
	test_num = "MPE"
	env_name = "crossing_team_greedy"
	experiment_type = "prd_above_threshold" # shared, prd_above_threshold

	dictionary = {
			"iteration": seed,
			"update_type": "ppo", # [ppo, a2c]
			"attention_type": "semi-hard", # [semi-hard, soft]
			"grad_clip_critic": 10.0,
			"grad_clip_actor": 10.0,
			"device": "gpu",
			"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
			"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
			"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_clip": 0.05,
			"value_clip": 0.05,
			"n_epochs": 5,
			"update_ppo_agent": 1, # update ppo agent after every 'update_ppo_agent' episodes
			"env": env_name, 
			"test_num":test_num,
			"extension":extension,
			"value_lr": 1e-3,
			"policy_lr": 7e-4,
			"entropy_pen": 0.0,
			"gamma": 0.99, 
			"gae_lambda": 0.95,
			"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
			"select_above_threshold": 0.05, # [0.05: 'soft', 0.0: 'semi-hard' --> attention_type]
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
			"max_episodes": 80000,
			"max_time_steps": 50,
			"experiment_type": experiment_type,
		}
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MultiAgent(env,dictionary)
	ma_controller.run()