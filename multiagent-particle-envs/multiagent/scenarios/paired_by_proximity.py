import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		self.num_agents = 4
		self.num_landmarks = 4
		self.proximity_radius = 0.4
		print("NUMBER OF AGENTS:",self.num_agents)
		print("NUMBER OF LANDMARKS:",self.num_landmarks)
		world.collaborative = True

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

		base_color_agent = np.array([0.45, 0.45, 0.85])
		base_color_landmark = np.array([0.1, 0.1, 0.1])

		for i in range(int(self.num_agents/2)):
			world.agents[i].color = base_color_agent + i/self.num_agents
			world.agents[self.num_agents - 1 - i].color = base_color_agent + i/self.num_agents

		for i in range(int(self.num_landmarks/2)):
			world.landmarks[i].color = base_color_landmark + i/self.num_landmarks
			world.landmarks[self.num_landmarks - 1 - i].color = base_color_landmark + i/self.num_landmarks


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
		return True if dist < self.proximity_radius else False


	def reward_by_proximity(self, agent, world):
		my_index = int(agent.name[-1])
		agent_pairing = []

		my_dist_from_goal = np.sqrt(np.sum(np.square(world.agents[my_index].state.p_pos - world.landmarks[my_index].state.p_pos)))
		
		rew = -my_dist_from_goal
		if world.agents[my_index].collide:
			for a in world.agents:
				if self.is_collision(a, world.agents[my_index]):
					rew -= 1
					agent_pairing.append(1)
				else:
					agent_pairing.append(0)
		
		return rew,agent_pairing


	def observation(self, agent, world):

		curr_agent_index = world.agents.index(agent)

		current_agent_critic = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]
		current_agent_actor = [agent.state.p_pos,agent.state.p_vel,world.landmarks[curr_agent_index].state.p_pos]
		other_agents_actor = []

		for other_agent in world.agents:
			if other_agent is agent:
			  continue
			other_agents_actor.append(other_agent.state.p_pos-agent.state.p_pos)
			other_agents_actor.append(other_agent.state.p_vel-agent.state.p_vel)

		return np.concatenate(current_agent_critic),np.concatenate(current_agent_actor+other_agents_actor)


	def isFinished(self,agent,world):
		index = len(world.agents)-int(agent.name[-1])-1
		dist = np.sqrt(np.sum(np.square(world.agents[index].state.p_pos - world.landmarks[index].state.p_pos)))
		if dist<0.075:
			return True
		return False
		
