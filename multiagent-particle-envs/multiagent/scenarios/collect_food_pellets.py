import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2

		self.threshold_dist = 0.1
		# Number of agents
		self.num_agents = 4
		# Food Pellets
		self.num_landmarks = 4
		# Number of teams
		self.num_teams = 2

		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		print("NUMBER OF TEAMS:",self.num_teams)

		world.collaborative = True


		# add agents
		world.agents = [Agent() for i in range(self.num_agents)]
		# add landmarks
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]


		# agent attributes
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = True
			agent.silent = True
			agent.size = 0.1

		# landmark attributes
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = False
			landmark.movable = False


		self.agent_team = []
		self.landmark_team = []
		num_agent_in_team = self.num_agents / self.num_teams
		num_landmark_in_team = self.num_landmarks / self.num_teams

		for i in range(self.num_teams-1):
			self.agent_team.append([world.agents[i*num_agent_in_team:(i+1)*num_agent_in_team]])
			self.landmark_team.append([world.landmarks[i*num_landmark_in_team:(i+1)*num_landmark_in_team]])


		# make initial conditions
		self.reset_world(world)
		return world

	def reset_world(self, world):

		# assign colors according to team
		team_colors = [np.random.rand(3,) for i in range(self.num_teams)]

		for i in range(self.num_teams):
			for agent in self.agent_team[i]:
				agent.color = team_colors[i]
				agent.team_id = i
			for landmark in self.landmark_team[i]:
				landmark.color = team_colors[i]
				landmark.team_id = i


		# set random initial states for landmark and agents
		for agent in world.agents:
			agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.c = np.zeros(world.dim_c)

		for i, landmark in enumerate(world.landmarks):
			landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			landmark.state.p_vel = np.zeros(world.dim_p)

	def benchmark_data(self, agent, world):
		rew = 0
		collisions = 0
		occupied_landmarks = 0
		min_dists = 0
		for l in world.landmarks:
			dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
			min_dists += min(dists)
			rew -= min(dists)
			if min(dists) < 0.1:
				occupied_landmarks += 1
		if agent.collide:
			for a in world.agents:
				if self.is_collision(a, agent):
					rew -= 1
					collisions += 1
		return (rew, collisions, min_dists, occupied_landmarks)


	def is_collision(self, agent1, agent2):
		if agent1.name == agent2.name:
			return False
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		dist_min = agent1.size + agent2.size
		return True if dist < dist_min else False


	def reward_paired_agents(self, agent, world):

		team_id = agent.team_id

		for l in self.landmark_team[team_id]:
			dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in self.agent_team[team_id]]
			rew -= min(dists)

		# COLLISION with Team only
		if agent.collide:
			for a in self.agent_team[team_id]:
				if self.is_collision(a, agent):
					rew -= 0.04


		# # COLLISION with all teams
		# if agent.collide:
		# 	for a in world.agents:
		# 		if self.is_collision(a, agent):
		# 			rew -= 0.04
		
		return rew


	def observation(self, agent, world):
		team_id = agent.team_id

		current_agent_critic = [agent.state.p_pos, agent.state.p_vel,agent.team_id]
		current_agent_actor = [agent.state.p_pos, agent.state.p_vel,agent.team_id]

		other_agents_actor = []

		for other_agent in world.agents:
			if other_agent is agent:
			  continue
			other_agents_actor.append(other_agent.state.p_pos-agent.state.p_pos)
			other_agents_actor.append(other_agent.state.p_vel-agent.state.p_vel)
			other_agents_actor.append(other_agent.team_id)

		current_agent_actor = current_agent_actor+other_agents_actor

		for landmark in world.landmarks:
			current_agent_critic.append(landmark.state.p_pos)
			current_agent_critic.append(landmark.team_id)
			current_agent_actor.append(landmark.state.p_pos-agent.state.p_pos)
			current_agent_actor.append(landmark.team_id)



		return np.concatenate(current_agent_critic),np.concatenate(current_agent_actor)


	def isFinished(self,agent,world):
		team_id = agent.team_id
		dist_from_goal = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in self.landmark_team[team_id]]
		if min(dist_from_goal)<self.threshold_dist:
			return True
		return False
		
