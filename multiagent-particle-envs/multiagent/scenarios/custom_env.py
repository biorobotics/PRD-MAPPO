import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
	def make_world(self):
		world = World()
		# set any world properties first
		# world.dim_c = 2
		num_agents = 4
		num_landmarks = 4
		print("NUMBER OF AGENTS:",num_agents)
		print("NUMBER OF LANDMARKS:",num_landmarks)
		self.num_landmarks = num_landmarks
		world.collaborative = True

		# add agents
		world.agents = [Agent() for i in range(num_agents)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i
			agent.collide = True
			agent.silent = True
			agent.size = 0.15 #was 0.15
		# add landmarks
		world.landmarks = [Landmark() for i in range(num_landmarks)]
		for i, landmark in enumerate(world.landmarks):
			landmark.name = 'landmark %d' % i
			landmark.collide = False
			landmark.movable = False
		# make initial conditions
		self.reset_world(world)
		return world

	def reset_world(self, world):

		num_agents = len(world.agents)
		num_landmarks = len(world.landmarks)

		base_color_agent = np.array([0.45, 0.45, 0.85])
		base_color_landmark = np.array([0.1, 0.1, 0.1])

		for i in range(int(num_agents/2)):
			world.agents[i].color = base_color_agent + i/num_agents
			world.agents[num_agents - 1 - i].color = base_color_agent + i/num_agents

		for i in range(int(num_landmarks/2)):
			world.landmarks[i].color = base_color_landmark + i/num_landmarks
			world.landmarks[num_landmarks - 1 - i].color = base_color_landmark + i/num_landmarks

		# # random properties for agents
		# for i, agent in enumerate(world.agents):
		# 	agent.color = np.array([0.35, 0.35, 0.85])
		# # random properties for landmarks
		# for i, landmark in enumerate(world.landmarks):
		# 	landmark.color = np.array([0.25, 0.25, 0.25])


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


	def reward_agent_centric_paired(self, agent, world):
		index = len(world.agents)-int(agent.name[-1])-1

		dist = np.sqrt(np.sum(np.square(world.agents[index].state.p_pos - world.landmarks[index].state.p_pos)))
		rew = -dist
		if world.agents[index].collide:
			for a in world.agents:
				if self.is_collision(a, world.agents[index]):
					rew -= 1
		return rew


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
		if dist<0.1:
			return True
		return False
		
