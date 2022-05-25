import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
# import webcolors


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		self.num_agents = 4
		self.num_landmarks = 4
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		world.collaborative = True
		self.col_pen = .1
		world.col_pen = self.col_pen

		self.num_teams = self.num_agents//2
		self.teams = {}
		for i in range(0,self.num_agents,2):
			self.teams[i] = i+1
			self.teams[i+1] = i

		self.timer = 0
		self.time_on_goal = [float('inf') for i in range(self.num_agents)]
		print("NUMBER OF TEAMS", self.num_teams)
		print("TEAMS", self.teams)

		# add agents
		world.agents = [Agent() for i in range(self.num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			# if i%2 == 0:
			# 	agent.name = 'fast_agent %d' % i
			# else:
			# 	agent.name = 'slow_agent %d' % i
			agent.collide = False
			agent.silent = True
			agent.size = 0.1 #was 0.15
			agent.prevDistance = None #0.0
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
				dist_min = (agent.size + other_agent.size) * 1.5
				if dist < dist_min:
					return True 

			return False

		elif landmark is not None and landmark_list is not None:
			for other_landmark in landmark_list:
				if landmark.name == other_landmark.name:
					continue
				delta_pos = landmark.state.p_pos - other_landmark.state.p_pos
				dist = np.sqrt(np.sum(np.square(delta_pos)))
				dist_min = (landmark.size + other_landmark.size) * 1.5
				if dist < dist_min:
					return True 

			return False

	def reset_world(self, world):
		color_choice = 2*[np.array([255,0,0]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,0,0]), np.array([128,0,0]), np.array([0,128,0]), np.array([0,0,128]), np.array([128,128,128]), np.array([128,0,128]), np.array([128,128,0])]

		for i in range(self.num_agents):
			# rgb = np.random.uniform(-1,1,3)
			# rgb = np.random.randint(0,255,3)
			# print(rgb)
			# world.agents[i].color = rgb
			# world.landmarks[i].color = rgb
			world.agents[i].color = color_choice[i]
			world.landmarks[i].color = color_choice[i]
			# print("AGENT", world.agents[i].name[-1], ":", webcolors.rgb_to_name((color_choice[i][0],color_choice[i][1],color_choice[i][2])))

		agent_list = []
		# set random initial states
		for agent in world.agents:
			agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

			while self.check_collision_before_spawning(agent, None, agent_list, None):
				agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.c = np.zeros(world.dim_c)
			agent.prevDistance = None # 0.0
			agent_list.append(agent)

		landmark_list = []
		for landmark in world.landmarks:
			landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			
			while self.check_collision_before_spawning(None, landmark, None, landmark_list):
				landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			
			landmark.state.p_vel = np.zeros(world.dim_p)
			landmark_list.append(landmark)

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
		dist_min = (agent1.size + agent2.size) * 1.5
		return True if dist < dist_min else False


	def increment_timer(self):
		self.timer += 1

	def reward(self, agent, world):
		my_index = int(agent.name[-1])

		my_dist_from_goal = np.sqrt(np.sum(np.square(world.agents[my_index].state.p_pos - world.landmarks[my_index].state.p_pos)))

		
		if agent.prevDistance is not None:
			rew = agent.prevDistance - my_dist_from_goal
		else:
			rew = 0.0

		agent.prevDistance = my_dist_from_goal

		# if agent.collide:
		# for a in world.agents:
		# 	if self.is_collision(a, agent):
		# 		rew -= self.col_pen


		if my_dist_from_goal < 0.1 and self.time_on_goal[my_index] != float('inf'):
			self.time_on_goal[my_index] = self.timer

		# # SHARED COLLISION REWARD
		# for a in world.agents:
		# 	for o in world.agents:
		# 		if self.is_collision(a,o):
		# 			rew -=0.1/self.num_agents

		# COLLISION REWARD FOR OTHER AGENTS
		# for a in world.agents:
		# 	if a.name != agent.name:
		# 		for o in world.agents:
		# 			if o.name != agent.name:
		# 				if self.is_collision(a,o):
		# 					rew -= self.col_pen
		
		# if my_dist_from_goal > .1:
		# 	# add existance penalty
		# 	rew += -0.01


		partner_index = self.teams[my_index]
		# if agent i reaches before agent i+1, both get +1 on completion and if agent i+1 reaches before agent i, both get +2
		if self.time_on_goal[my_index] != float('inf') and self.time_on_goal[partner_index] != float('inf'): # current agent reached its goal
			if my_index == 0 and self.time_on_goal[partner_index] < self.time_on_goal[my_index]:
				rew+=5
			elif my_index == 0 and self.time_on_goal[partner_index] >= self.time_on_goal[my_index]:
				rew+=1

			elif my_index == 1 and self.time_on_goal[partner_index] < self.time_on_goal[my_index]:
				rew+=1
			elif my_index == 1 and self.time_on_goal[partner_index] >= self.time_on_goal[my_index]:
				rew+=5

			elif my_index == 2 and self.time_on_goal[partner_index] < self.time_on_goal[my_index]:
				rew+=5
			elif my_index == 2 and self.time_on_goal[partner_index] >= self.time_on_goal[my_index]:
				rew+=1

			elif my_index == 3 and self.time_on_goal[partner_index] < self.time_on_goal[my_index]:
				rew+=1
			elif my_index == 3 and self.time_on_goal[partner_index] >= self.time_on_goal[my_index]:
				rew+=5

			elif my_index == 4 and self.time_on_goal[partner_index] < self.time_on_goal[my_index]:
				rew+=5
			elif my_index == 4 and self.time_on_goal[partner_index] >= self.time_on_goal[my_index]:
				rew+=1

			elif my_index == 5 and self.time_on_goal[partner_index] < self.time_on_goal[my_index]:
				rew+=1
			elif my_index == 5 and self.time_on_goal[partner_index] >= self.time_on_goal[my_index]:
				rew+=10

			elif my_index == 6 and self.time_on_goal[partner_index] < self.time_on_goal[my_index]:
				rew+=5
			elif my_index == 6 and self.time_on_goal[partner_index] >= self.time_on_goal[my_index]:
				rew+=1

			elif my_index == 7 and self.time_on_goal[partner_index] < self.time_on_goal[my_index]:
				rew+=1
			elif my_index == 7 and self.time_on_goal[partner_index] >= self.time_on_goal[my_index]:
				rew+=5


		# increment time when last agent's rewards are processed
		if my_index == self.num_agents-1:
			self.increment_timer()


		return rew


	def observation(self, agent, world):

		curr_agent_index = world.agents.index(agent)
		
		current_agent_critic = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]
		
		current_agent_actor = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]
		# other_agents_actor = []

		# for other_agent in world.agents:
		# 	if other_agent is agent:
		# 	  continue
		# 	other_agents_actor.append(other_agent.state.p_pos-agent.state.p_pos)
		# 	other_agents_actor.append(other_agent.state.p_vel-agent.state.p_vel)

		# return np.concatenate(current_agent_critic),np.concatenate(current_agent_actor+other_agents_actor)
		return np.concatenate(current_agent_critic),np.concatenate(current_agent_actor)


	def isFinished(self,agent,world):
		index = int(agent.name[-1])
		dist = np.sqrt(np.sum(np.square(world.agents[index].state.p_pos - world.landmarks[index].state.p_pos)))
		if dist<0.1:
			return True
		return False
		
