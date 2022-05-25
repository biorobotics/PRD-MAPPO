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
		self.num_agents = 12
		self.num_landmarks = 12
		self.threshold_dist = 1e-1
		self.goal_reward = 1e-1
		self.pen_existence = 1e-2
		self.team_size = 4
		self.pen_collision = 0.1
		self.agent_size = 0.1
		self.landmark_size = 0.1
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		print("TEAM SIZE", self.team_size)
		world.collaborative = True

		# add agents
		world.agents = [Agent() for i in range(self.num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = False
			agent.silent = True
			agent.size = self.agent_size
			agent.prevDistance = None
			if i<self.team_size:
				agent.team_id = 1
			elif i>= self.team_size and i<2*self.team_size:
				agent.team_id = 2
			elif i>= 2*self.team_size and i<3*self.team_size:
				agent.team_id = 3
			elif i>= 3*self.team_size and i<4*self.team_size:
				agent.team_id = 4
			else:
				agent.team_id = 5
		# add landmarks
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = False
			landmark.movable = False
			if i<self.team_size:
				landmark.team_id = 1
			elif i>= self.team_size and i<2*self.team_size:
				landmark.team_id = 2
			elif i>= 2*self.team_size and i<3*self.team_size:
				landmark.team_id = 3
			elif i>= 3*self.team_size and i<4*self.team_size:
				landmark.team_id = 4
			else:
				landmark.team_id = 5
		# make initial conditions
		self.reset_world(world)
		return world


	def check_collision_before_spawning(self,agent,landmark,agent_list,landmark_list):
		if agent is not None and agent_list is not None:
			for other_agent in agent_list:
				if agent.name == other_agent.name or agent.team_id != other_agent.team_id:
					continue
				delta_pos = agent.state.p_pos - other_agent.state.p_pos
				dist = np.sqrt(np.sum(np.square(delta_pos)))
				dist_min = self.agent_size*3
				if dist < dist_min:
					return True 

			return False

		elif landmark is not None and landmark_list is not None:
			for other_landmark in landmark_list:
				if landmark.name == other_landmark.name or landmark.team_id != other_landmark.team_id:
					continue
				delta_pos = landmark.state.p_pos - other_landmark.state.p_pos
				dist = np.sqrt(np.sum(np.square(delta_pos)))
				dist_min = self.agent_size*3
				if dist < dist_min:
					return True 

			return False

	def reset_world(self, world):
		agent_list = []
		color_choice = [np.array([255,0,0]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,0,0]), np.array([255,0,255])]
		
		for i in range(self.num_agents):
			# rgb = np.random.uniform(-1,1,3)
			if i < self.team_size:
				world.agents[i].color = color_choice[0]
				world.landmarks[i].color = color_choice[0]
			elif i>=self.team_size and i<2*self.team_size:
				world.agents[i].color = color_choice[1]
				world.landmarks[i].color = color_choice[1]
			elif i>=2*self.team_size and i<3*self.team_size:
				world.agents[i].color = color_choice[2]
				world.landmarks[i].color = color_choice[2]
			elif i>=3*self.team_size and i<4*self.team_size:
				world.agents[i].color = color_choice[3]
				world.landmarks[i].color = color_choice[3]
			else:
				world.agents[i].color = color_choice[4]
				world.landmarks[i].color = color_choice[4]

			if i%self.team_size == 0:
				y = random.uniform(-1,1)
				x = -1
				world.agents[i].state.p_pos = np.array([x,y])
				world.landmarks[i].state.p_pos = np.array([-x,y])
				while self.check_collision_before_spawning(world.agents[i],None, agent_list,None):
					y = random.uniform(-1,1)
					world.agents[i].state.p_pos = np.array([x,y])
					world.landmarks[i].state.p_pos = np.array([-x,y])
				world.agents[i].direction = "y"
			elif i%self.team_size == 1:
				x = random.uniform(-1,1)
				y = -1
				world.agents[i].state.p_pos = np.array([x,y])
				world.landmarks[i].state.p_pos = np.array([x,-y])
				while self.check_collision_before_spawning(world.agents[i],None, agent_list,None):
					x = random.uniform(-1,1)
					world.agents[i].state.p_pos = np.array([x,y])
					world.landmarks[i].state.p_pos = np.array([x,-y])
				world.agents[i].direction = "x"
			elif i%self.team_size == 2:
				y = random.uniform(-1,1)
				x = 1
				world.agents[i].state.p_pos = np.array([x,y])
				world.landmarks[i].state.p_pos = np.array([-x,y])
				while self.check_collision_before_spawning(world.agents[i],None, agent_list,None):
					y = random.uniform(-1,1)
					world.agents[i].state.p_pos = np.array([x,y])
					world.landmarks[i].state.p_pos = np.array([-x,y])
				world.agents[i].direction = "-y"
			elif i%self.team_size == 3:
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
		if agent1.name == agent2.name or agent1.team_id != agent2.team_id:
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
			rew = (agent.prevDistance - agent_dist_from_goal)

		agent.prevDistance = agent_dist_from_goal

		# rew = -agent_dist_from_goal/10.0

		collision_count = 0
		for other_agent in world.agents:
			if self.is_collision(agent, other_agent):
				rew -= self.pen_collision
				collision_count += 1

		# on reaching goal we reward the agent
		goal_reached = 0
		if agent_dist_from_goal<self.threshold_dist:
		# 	# rew += self.goal_reward
			# agent.goal_reached = True
			goal_reached = 1
		# else:
		# 	agent.goal_reached = False
		# 	rew -= self.pen_existence
			

		return rew, collision_count, goal_reached


	def observation(self, agent, world):
		if agent.state.p_pos[0]>1.0:
			agent.state.p_pos[0] = 1.0
		if agent.state.p_pos[0]<-1.0:
			agent.state.p_pos[0] = -1.0
		if agent.state.p_pos[1]<-1.0:
			agent.state.p_pos[1] = -1.0
		if agent.state.p_pos[1]>1.0:
			agent.state.p_pos[1] = 1.0

		curr_agent_index = world.agents.index(agent)
		# team = [0 for i in range(self.num_agents//self.team_size)]
		# team[agent.team_id-1] = 1
		# team = np.array(team)
		team = np.array([agent.team_id])
		current_agent_critic = [agent.state.p_pos, agent.state.p_vel, team, world.landmarks[curr_agent_index].state.p_pos]
		current_agent_actor = [agent.state.p_pos, agent.state.p_vel, team, world.landmarks[curr_agent_index].state.p_pos]
		for other_agent in world.agents:
			if other_agent.name == agent.name:
				continue
			# relative pose, velocity wrt current_agent and team id of other agent 
			current_agent_actor.extend([other_agent.state.p_pos-agent.state.p_pos,other_agent.state.p_vel-agent.state.p_vel,np.array([other_agent.team_id])])
		
		return np.concatenate(current_agent_critic), np.concatenate(current_agent_actor)


	def isFinished(self,agent,world):
		for other_agent in world.agents:
			index = world.agents.index(other_agent)
			dist = np.sqrt(np.sum(np.square(other_agent.state.p_pos - world.landmarks[index].state.p_pos)))
			if dist>self.threshold_dist:
				return False
		return True
		
