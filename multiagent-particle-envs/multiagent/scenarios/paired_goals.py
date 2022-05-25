import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from itertools import cycle



class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		self.threshold_dist = 0.1
		self.num_agents = 4
		self.num_landmarks = 4

		self.exist_pen = -0.001
		self.cum_pen = [0]*self.num_agents

		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		world.collaborative = True


		# add agents
		world.agents = [Agent() for i in range(self.num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = True
			agent.silent = True
			agent.size = 0.1

		# Pairing landmarks i with N-i
		# add landmarks
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = False
			landmark.movable = False


		# make initial conditions
		self.reset_world(world)
		return world

	def reset_world(self, world):

		base_color_agent = np.array([0.45, 0.45, 0.85])
		base_color_landmark = np.array([0.1, 0.1, 0.1])

		for i in range(int(self.num_agents)):
			world.agents[i].color = base_color_agent + (i*self.num_agents)/10.0

		for i in range(int(self.num_landmarks/2)):
			world.landmarks[i].color = base_color_landmark + (i*self.num_landmarks)/10.0
			world.landmarks[self.num_landmarks - 1 - i].color = base_color_landmark + (i*self.num_agents)/10.0


		# set random initial states
		for agent in world.agents:
			agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.c = np.zeros(world.dim_c)

		for i, landmark in enumerate(world.landmarks):
			landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			landmark.state.p_vel = np.zeros(world.dim_p)

		self.cum_pen = [0]*self.num_agents

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

		# penalty of existence till timestep t
		# agent_id = int(agent.name[-1])
		# rew = self.cum_pen[agent_id]
		# self.cum_pen[agent_id] += self.exist_pen

		rew = -0.01
		# calculating distance from goal positions
		dist_from_landmark = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
		
		if min(dist_from_landmark) < self.threshold_dist:
			# First of the landmark pair
			if np.argmin(dist_from_landmark) < self.num_landmarks/2:
				rew += 1.0
			else:
				rew += 0.5

		# reward due to implicit partner being on the other paired goal position
		if min(dist_from_landmark) < self.threshold_dist and np.argmin(dist_from_landmark) < self.num_landmarks/2:
			# check if the other paired goal location is occupied or not
			paired_landmark_index = self.num_landmarks - 1 - np.argmin(dist_from_landmark)
			# calculate distance of paired landmark with other agents in the environment
			dist_from_paired_landmark = [np.sqrt(np.sum(np.square(world.landmarks[paired_landmark_index].state.p_pos - other_agent.state.p_pos))) for other_agent in world.agents]
			# should not be the same agent and should be at the landmark
			if min(dist_from_paired_landmark) < self.threshold_dist and np.argmin(dist_from_paired_landmark) != int(agent.name[-1]):
				rew += 5.0

		# # COLLISION
		# if agent.collide:
		# 	for a in world.agents:
		# 		if self.is_collision(a, agent):
		# 			rew -= 0.04
		
		return rew


	def observation(self, agent, world):

		current_agent_critic = [agent.state.p_pos,agent.state.p_vel]

		
		current_agent_actor = [agent.state.p_pos,agent.state.p_vel]
		other_agents_actor = []

		for other_agent in world.agents:
			if other_agent is agent:
			  continue
			other_agents_actor.append(other_agent.state.p_pos-agent.state.p_pos)
			other_agents_actor.append(other_agent.state.p_vel-agent.state.p_vel)

		current_agent_actor = current_agent_actor+other_agents_actor

		for landmark in world.landmarks:
			current_agent_critic.append(landmark.state.p_pos)
			current_agent_actor.append(landmark.state.p_pos - agent.state.p_pos)


		return np.concatenate(current_agent_critic),np.concatenate(current_agent_actor)


	def isFinished(self,agent,world):
		dist_from_landmark = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
		if min(dist_from_landmark)<self.threshold_dist:
			return True
		return False
		
