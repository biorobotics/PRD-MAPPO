from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math

class RolloutBuffer:
	def __init__(self):
		self.state_agents = []
		self.state_opponents = []
		self.probs = []
		self.logprobs = []
		self.actions = []
		self.one_hot_actions = []


		self.rewards = []
		self.dones = []

		self.values = []
		self.qvalues = []
	

	def clear(self):
		del self.actions[:]
		del self.state_agents[:]
		del self.state_opponents[:]
		del self.probs[:]
		del self.one_hot_actions[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.dones[:]
		del self.values[:]
		del self.qvalues[:]

class Policy(nn.Module):
	def __init__(self, obs_input_dim, num_actions, num_agents, device):
		super(Policy, self).__init__()

		self.name = "MLP Policy"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.Policy_MLP = nn.Sequential(
			nn.Linear(obs_input_dim, 256),
			nn.Tanh(),
			nn.Linear(256, 64),
			nn.Tanh(),
			nn.Linear(64, num_actions),
			nn.Softmax(dim=-1)
			)

		self.reset_parameters()

	def reset_parameters(self):
		gain = nn.init.calculate_gain('tanh')
		gain_last_layer = nn.init.calculate_gain('tanh', 0.01)

		nn.init.orthogonal_(self.Policy_MLP[0].weight, gain=gain)
		nn.init.orthogonal_(self.Policy_MLP[2].weight, gain=gain)
		nn.init.orthogonal_(self.Policy_MLP[4].weight, gain=gain_last_layer)


	def forward(self, state_agents, state_opponents):
		state_agents_aug = torch.stack([torch.roll(state_agents,-i,1) for i in range(self.num_agents)], dim=0).transpose(1,0).reshape(state_agents.shape[0],self.num_agents,-1)
		observations = torch.cat([state_agents_aug, state_opponents], dim=-1)
		return self.Policy_MLP(observations)


# using Q network of MAAC
class Q_network(nn.Module):
	def __init__(self, obs_input_dim, num_agents, num_actions, attention_type, device):
		super(Q_network, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.attention_type = attention_type

		obs_output_dim = 128
		obs_act_input_dim = obs_input_dim+self.num_actions
		obs_act_output_dim = 128
		curr_agent_output_dim = 128

		self.state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, 256, bias=True), 
			nn.Tanh()
			)
		self.key = nn.Linear(256, obs_output_dim, bias=True)
		self.query = nn.Linear(256, obs_output_dim, bias=True)
		if "semi-hard" in self.attention_type:
			self.hard_attention = nn.Sequential(
				nn.Linear(obs_output_dim*2, 64),
				nn.Tanh(),
				nn.Linear(64, 2)
				)
		
		self.state_act_embed = nn.Sequential(
			nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=True), 
			nn.Tanh()
			)
		self.attention_value = nn.Sequential(
			nn.Linear(obs_act_output_dim, 128, bias=True), 
			nn.Tanh()
			)

		self.curr_agent_state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, curr_agent_output_dim, bias=True), 
			nn.Tanh()
			)

		# dimesion of key
		self.d_k = obs_output_dim

		# ********************************************************************************************************

		# ********************************************************************************************************
		final_input_dim = 128 + curr_agent_output_dim
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.Tanh(),
			nn.Linear(64, self.num_actions, bias=True)
			)
		
		# ********************************************************************************************************
		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('tanh')

		nn.init.orthogonal_(self.state_embed[0].weight, gain=gain)
		nn.init.orthogonal_(self.state_act_embed[0].weight, gain=gain)

		nn.init.orthogonal_(self.key.weight)
		nn.init.orthogonal_(self.query.weight)
		if "semi-hard" in self.attention_type:
			nn.init.orthogonal_(self.hard_attention[0].weight, gain=gain)
			nn.init.orthogonal_(self.hard_attention[2].weight, gain=gain)
		nn.init.orthogonal_(self.attention_value[0].weight)

		nn.init.orthogonal_(self.curr_agent_state_embed[0].weight, gain=gain)

		nn.init.orthogonal_(self.final_value_layers[0].weight, gain=gain)
		nn.init.orthogonal_(self.final_value_layers[2].weight, gain=gain)


	def remove_self_loops(self, states_key):
		ret_states_keys = torch.zeros(states_key.shape[0],self.num_agents,self.num_agents-1,states_key.shape[-1])
		for i in range(self.num_agents):
			if i == 0:
				red_state = states_key[:,i,i+1:]
			elif i == self.num_agents-1:
				red_state = states_key[:,i,:i]
			else:
				red_state = torch.cat([states_key[:,i,:i],states_key[:,i,i+1:]], dim=-2)

			ret_states_keys[:,i] = red_state

		return ret_states_keys.to(self.device)

	def weight_assignment(self,weights):
		weights_new = torch.zeros(weights.shape[0], self.num_agents, self.num_agents).to(self.device)
		one = torch.ones(weights.shape[0],1).to(self.device)
		for i in range(self.num_agents):
			if i == 0:
				weight_vec = torch.cat([one,weights[:,i,:]], dim=-1)
			elif i == self.num_agents-1:
				weight_vec = torch.cat([weights[:,i,:],one], dim=-1)
			else:
				weight_vec = torch.cat([weights[:,i,:i],one,weights[:,i,i:]], dim=-1)

			weights_new[:,i] = weight_vec

		return weights_new


	def forward(self, state_agents, state_opponents, policies, actions):
		# states = torch.cat([state_agents, state_opponents.unsqueeze(1).repeat(1,self.num_agents,1)], dim=-1)
		states = torch.cat([state_agents, state_opponents], dim=-1)
		states_query = states.unsqueeze(-2)
		states_key = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
		actions_ = actions.unsqueeze(1).repeat(1,self.num_agents,1,1)

		states_key = self.remove_self_loops(states_key)
		actions_ = self.remove_self_loops(actions_)

		obs_actions = torch.cat([states_key,actions_],dim=-1)

		# EMBED STATES QUERY
		states_query_embed = self.state_embed(states_query)
		# EMBED STATES QUERY
		states_key_embed = self.state_embed(states_key)
		# KEYS
		key_obs = self.key(states_key_embed)
		# QUERIES
		query_obs = self.query(states_query_embed)
		# WEIGHT
		if 'semi-hard' in self.attention_type:
			query_vector = torch.cat([query_obs.repeat(1,1,self.num_agents-1,1), key_obs], dim=-1)
			hard_weights = nn.functional.gumbel_softmax(self.hard_attention(query_vector), hard=True, dim=-1)
			prop = hard_weights[:,:,:,1]
			hard_score = -10000*(1-prop) + prop
			score = (torch.matmul(query_obs,key_obs.transpose(2,3))/math.sqrt(self.d_k)) + hard_score.unsqueeze(-2)
			weight = F.softmax(score ,dim=-1)
			weights = self.weight_assignment(weight.squeeze(-2))
		else:
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(2,3))/math.sqrt(self.d_k),dim=-1)
			weights = self.weight_assignment(weight.squeeze(-2))

		# EMBED STATE ACTION POLICY
		obs_actions_embed = self.state_act_embed(obs_actions)
		attention_values = self.attention_value(obs_actions_embed)
		node_features = torch.matmul(weight, attention_values)

		curr_agent_state_embed = self.curr_agent_state_embed(states)
		curr_agent_node_features = torch.cat([curr_agent_state_embed, node_features.squeeze(-2)], dim=-1)
		
		Q_value = self.final_value_layers(curr_agent_node_features)

		Value = torch.matmul(Q_value,policies.transpose(1,2))

		Q_value = torch.sum(actions*Q_value, dim=-1).unsqueeze(-1)

		return Value, Q_value, weights