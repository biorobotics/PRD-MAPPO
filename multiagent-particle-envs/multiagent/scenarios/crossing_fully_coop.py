import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import webcolors
import math
import random


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		self.num_agents = 8
		self.num_landmarks = 8
		self.pen_existence = 0.1
		self.pen_collision = 0.1
		self.agent_size = 0.15 # agent size = 0.1 (16 Agents)/ agent size = 0.15
		self.landmark_size = 0.1
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		world.collaborative = True

		# add agents
		world.agents = [Agent() for i in range(self.num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = False
			agent.silent = True
			agent.size = self.agent_size
			agent.prevDistance = None
		# add landmarks
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = False
			landmark.movable = False
		# make initial conditions
		self.reset_world(world)
		return world


	def check_collision_before_spawning(self,agent,landmark,agent_list,landmark_list):

		if agent is not None and agent_list is not None:
			for other_agent in agent_list:
				if agent.name == other_agent.name:
					continue
				delta_pos = agent.state.p_pos - other_agent.state.p_pos
				dist = np.sqrt(np.sum(np.square(delta_pos)))
				dist_min = (agent.size + other_agent.size)
				if dist < dist_min:
					return True 

			return False

		elif landmark is not None and landmark_list is not None:
			for other_landmark in landmark_list:
				if landmark.name == other_landmark.name:
					continue
				delta_pos = landmark.state.p_pos - other_landmark.state.p_pos
				dist = np.sqrt(np.sum(np.square(delta_pos)))
				dist_min = self.agent_size*2
				if dist < dist_min:
					return True 

			return False

	def reset_world(self, world):
		agent_list = []
		for i in range(self.num_agents):
			rgb = np.random.uniform(-1,1,3)
			world.agents[i].color = rgb
			world.landmarks[i].color = rgb

			if i%4 == 0:
				y = random.uniform(-1,1)
				x = -1
				world.agents[i].state.p_pos = np.array([x,y])
				world.landmarks[i].state.p_pos = np.array([-x,y])
				while self.check_collision_before_spawning(world.agents[i],None, agent_list,None):
					y = random.uniform(-1,1)
					world.agents[i].state.p_pos = np.array([x,y])
					world.landmarks[i].state.p_pos = np.array([-x,y])
				world.agents[i].direction = "y"
			elif i%4 == 1:
				x = random.uniform(-1,1)
				y = -1
				world.agents[i].state.p_pos = np.array([x,y])
				world.landmarks[i].state.p_pos = np.array([x,-y])
				while self.check_collision_before_spawning(world.agents[i],None, agent_list,None):
					x = random.uniform(-1,1)
					world.agents[i].state.p_pos = np.array([x,y])
					world.landmarks[i].state.p_pos = np.array([x,-y])
				world.agents[i].direction = "x"
			elif i%4 == 2:
				y = random.uniform(-1,1)
				x = 1
				world.agents[i].state.p_pos = np.array([x,y])
				world.landmarks[i].state.p_pos = np.array([-x,y])
				while self.check_collision_before_spawning(world.agents[i],None, agent_list,None):
					y = random.uniform(-1,1)
					world.agents[i].state.p_pos = np.array([x,y])
					world.landmarks[i].state.p_pos = np.array([-x,y])
				world.agents[i].direction = "-y"
			elif i%4 == 3:
				x = random.uniform(-1,1)
				y = 1
				world.agents[i].state.p_pos = np.array([x,y])
				world.landmarks[i].state.p_pos = np.array([x,-y])
				while self.check_collision_before_spawning(world.agents[i],None, agent_list,None):
					x = random.uniform(-1,1)
					world.agents[i].state.p_pos = np.array([x,y])
					world.landmarks[i].state.p_pos = np.array([x,-y])
				world.agents[i].direction = "-x"

			agent_list.append(world.agents[i])

			world.agents[i].state.p_vel = np.zeros(world.dim_p)
			world.agents[i].state.c = np.zeros(world.dim_c)
			world.agents[i].prevDistance = None
			world.landmarks[i].state.p_vel = np.zeros(world.dim_p)



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
		dist_min = (agent1.size + agent2.size) #+ 0.1 # adding +1 for 16 agent case because we decreased agent size
		return True if dist < dist_min else False


	def reward(self, agent, world):
		my_index = int(agent.name[-1])
		
		agent_dist_from_goal = np.sqrt(np.sum(np.square(world.agents[my_index].state.p_pos - world.landmarks[my_index].state.p_pos)))

		if agent.prevDistance is None:
			rew = 0
		else:
			rew = agent.prevDistance - agent_dist_from_goal

		agent.prevDistance = agent_dist_from_goal

		# COLLISION REWARD FOR OTHER AGENTS
		for a in world.agents:
			if a.name != agent.name:
				for o in world.agents:
					if o.name != agent.name:
						if self.is_collision(a,o):
							rew -= self.pen_collision/2

		collision_count = 0
		for other_agent in world.agents:
			if self.is_collision(agent, other_agent):
				collision_count += 1
		
		return rew, collision_count


	def observation(self, agent, world):
		curr_agent_index = world.agents.index(agent)
		current_agent_critic = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]
		current_agent_actor = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]

		return np.concatenate(current_agent_critic), np.concatenate(current_agent_actor)


	def isFinished(self,agent,world):
		index = world.agents.index(agent)
		dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[index].state.p_pos)))
		if dist<0.1:
			return True
		return False
		
