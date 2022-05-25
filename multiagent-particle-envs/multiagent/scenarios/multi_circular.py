import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import webcolors
import math


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		self.num_agents = 8
		self.num_landmarks = 8
		self.num_circles = 2
		self.num_agents_per_circle = self.num_agents//self.num_circles # keeping it uniform (try to make it a perfectly divisible)
		self.col_pen = 0.1
		world.col_pen = self.col_pen
		print('COL PEN: ', self.col_pen)
		self.existence_pen = 0.0 #0.01
		world.existence_pen = self.existence_pen
		print('existence PEN: ', self.existence_pen)
		self.radius_circle = {1: 1, 2: 0.4, 3: 0.4, 4: 0.15} #(2/(self.num_circles*2))
		self.centers = {1: [(0.0,0.0)], 2:[(-0.5,0.0), (0.5,0.0)], 3:[(-0.5,-0.5), (0.5,-0.5), (0.0,0.5)], 4:[(-0.5,-0.5), (-0.5,0.5), (0.5, -0.5), (0.5, 0.5)]}#[(-0.5,0.0), (0.5,0.0)]
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		print("NUMBER OF CIRCLES:", self.num_circles)
		print("NUMBER OF AGENTS PER CIRCLE:", self.num_agents_per_circle)
		world.collaborative = True

		# add agents
		agent_size = .1
		world.agent_size = agent_size
		world.agents = [Agent() for i in range(self.num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = False
			agent.silent = True
			agent.size = agent_size
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


	def check_collision_before_spawning(self,agent,agent_list):

		if agent is not None and agent_list is not None:
			for other_agent in agent_list:
				if agent.name == other_agent.name:
					continue
				delta_pos = agent.state.p_pos - other_agent.state.p_pos
				dist = np.sqrt(np.sum(np.square(delta_pos)))
				dist_min = (agent.size + other_agent.size)
				if dist < dist_min:
					# print("COLLISION WHILE SPAWNING")
					return True 

			return False


	def reset_world(self, world):
		color_choice = [np.array([255,0,0]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,0,0]), np.array([128,0,0]), np.array([0,128,0]), np.array([0,0,128]), np.array([128,0,128]), np.array([128,128,0]), np.array([128,128,128])]
		# AGENT 0 : red
		# AGENT 1 : lime
		# AGENT 2 : blue
		# AGENT 3 : black

		# base_color = np.array([0.1, 0.1, 0.1])

		agent_list = []
		landmark_list = []

		for i in range(self.num_agents):
			rgb = np.random.uniform(-1,1,3)
			# world.agents[i].color = rgb
			# world.landmarks[i].color = rgb
			world.agents[i].color = color_choice[i]
			world.landmarks[i].color = color_choice[i]
			print("AGENT", world.agents[i].name[-1], ":", webcolors.rgb_to_name((color_choice[i][0],color_choice[i][1],color_choice[i][2])))

		# set random initial states
		agent_list = []
		radius = self.radius_circle[self.num_circles]
		start_agent_index = 0
		end_agent_index = self.num_agents_per_circle
		for center_index, center in enumerate(self.centers[self.num_circles]):
			for agent in world.agents[start_agent_index:end_agent_index]:
				theta = np.random.uniform(-math.pi, math.pi)
				x = center[0] + radius*math.cos(theta)
				y = center[1] + radius*math.sin(theta)
				agent.state.p_pos = np.array([x,y])
				x_g = center[0] + radius*math.cos(theta+math.pi)
				y_g = center[1] + radius*math.sin(theta+math.pi)
				world.landmarks[int(agent.name[-1])].state.p_pos = np.array([x_g,y_g])

				while self.check_collision_before_spawning(agent, agent_list):
					theta = np.random.uniform(-math.pi, math.pi)
					x = center[0] + radius*math.cos(theta)
					y = center[1] + radius*math.sin(theta)
					agent.state.p_pos = np.array([x,y])
					x_g = center[0] + radius*math.cos(theta+math.pi)
					y_g = center[1] + radius*math.sin(theta+math.pi)
					world.landmarks[int(agent.name[-1])].state.p_pos = np.array([x_g,y_g])

				agent.state.p_vel = np.zeros(world.dim_p)
				agent.state.c = np.zeros(world.dim_c)
				agent.prevDistance = None

				agent_list.append(agent)

			start_agent_index += self.num_agents_per_circle

			if (center_index == len(self.centers[self.num_circles]) - 2) and (end_agent_index+self.num_agents_per_circle < len(world.agents)):
				end_agent_index += len(world.agents) - start_agent_index
			elif end_agent_index+self.num_agents_per_circle < len(world.agents):
				end_agent_index += self.num_agents_per_circle
			else:
				end_agent_index += len(world.agents) - start_agent_index



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


	def reward(self, agent, world):

		my_index = int(agent.name[-1])

		agent_dist_from_goal = np.sqrt(np.sum(np.square(world.agents[my_index].state.p_pos - world.landmarks[my_index].state.p_pos)))

		if agent.prevDistance is None:
			rew = 0
		else:
			rew = agent.prevDistance - agent_dist_from_goal

		agent.prevDistance = agent_dist_from_goal

		# penalise on collision
		# for a in world.agents:
		# 	if self.is_collision(a, agent):
		# 		rew -= self.col_pen

		# assert False

		# # SHARED COLLISION REWARD
		# for a in world.agents:
		# 	for o in world.agents:
		# 		if self.is_collision(a,o):
		# 			rew -=0.01

		# COLLISION REWARD FOR OTHER AGENTS
		for a in world.agents:
			if a.name != agent.name:
				for o in world.agents:
					if o.name != agent.name:
						if self.is_collision(a,o):
							# print(str(a.name) +' in collision with '+str(o.name)+'   would add pen to '+str(world.agents[my_index].name))
							rew -= self.col_pen/2 # divide by 2 so as not to overcount collisions

		# Penalty of existence
		# if agent_dist_from_goal > .1:
		# 	rew -= self.existence_pen

		
		return rew


	def observation(self, agent, world):

		curr_agent_index = world.agents.index(agent)

		current_agent_critic = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]
		
		
		current_agent_actor = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]

		return np.concatenate(current_agent_critic),np.concatenate(current_agent_actor)


	def isFinished(self,agent,world):
		index = int(agent.name[-1])
		dist = np.sqrt(np.sum(np.square(world.agents[index].state.p_pos - world.landmarks[index].state.p_pos)))
		if dist<0.1:
			return True
		return False