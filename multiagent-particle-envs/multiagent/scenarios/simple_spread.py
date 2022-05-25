import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		self.num_agents = 2
		self.num_landmarks = 2
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		world.collaborative = True

		self.threshold_dist = 0.1

		# add agents
		world.agents = [Agent() for i in range(self.num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = True
			agent.silent = True
			agent.size = 0.15 #was 0.15
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

		# random properties for agents
		for i, agent in enumerate(world.agents):
			agent.color = np.array([0.35, 0.35, 0.85])
		# random properties for landmarks
		for i, landmark in enumerate(world.landmarks):
			landmark.color = np.array([0.25, 0.25, 0.25])
		# set random initial states
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

	
	# with respect to landmarks
	def reward(self, agent, world):
		# Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
		index = int(agent.name[-1])
		dist_from_agent = [np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[index].state.p_pos))) for agent in world.agents]

		rew = -min(dist_from_agent)

		which_agent = np.argmin(dist_from_agent)

		# COLLISON
		# if agent.collide:
		# 	for a in world.agents:
		# 		if self.is_collision(a, agent):
		# 			rew -= 1

		return (rew, which_agent)

	def observation(self, agent, world):
		index = int(int(agent.name[-1]))

		current_agent_critic = [agent.state.p_pos,agent.state.p_vel]

		# Can do this as number of agents and landmarks are equal so with each agent obs, we return a landmark's pose
		goal_state = world.landmarks[index].state.p_pos
		
		current_agent_actor = [agent.state.p_pos,agent.state.p_vel]
		other_agents_actor = []

		for other_agent in world.agents:
			if other_agent is agent:
			  continue
			other_agents_actor.append(other_agent.state.p_pos-agent.state.p_pos)
			other_agents_actor.append(other_agent.state.p_vel-agent.state.p_vel)

		current_agent_actor = current_agent_actor+other_agents_actor

		for landmark in world.landmarks:
			current_agent_actor.append(landmark.state.p_pos - agent.state.p_pos)


		return np.concatenate(current_agent_critic), np.concatenate(current_agent_actor), goal_state


	

	def isFinished(self,agent,world):
		index = int(agent.name[-1])
		dist_from_agent = [np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[index].state.p_pos))) for agent in world.agents]
		if min(dist_from_agent)<self.threshold_dist:
			return True
		return False
		
		
