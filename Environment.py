import functools
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete,MultiDiscrete,Box
from pettingzoo import ParallelEnv,AECEnv
from pettingzoo.utils import parallel_to_aec
import gymnasium
from pettingzoo.utils import agent_selector, wrappers
from Agent import Agent

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "grid_world"}

    def __init__(self,render_mode=None):
        self.epi = 0
        self.steps = 0
        self.render_mode = render_mode

        self.theta = np.random.uniform(-np.pi,np.pi)
        self.length = 250
        self.xf = self.length*np.cos(self.theta)
        self.yf = self.length*np.sin(self.theta)

        self.agent0 = Agent(self.theta,self.xf,self.yf,0,0)
        self.agent1 = Agent(self.theta,self.xf,self.yf,0,0)
        self.agent2 = Agent(self.theta,self.xf,self.yf,0,0)
        self.num_agents_ = 3
        self.possible_agents = [f"agent_{i}" for i in range(self.num_agents_)]

    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent):
        return Box(low = -1, high = 1, shape=(3,), dtype = np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self,agent):
        return Box(low = -1, high = 1, shape = (1,), dtype = np.float32)

    def render(self):
        plt.scatter(0,0)
        plt.scatter(self.xf,self.yf)
        plt.plot([0,self.xf],[0,self.yf],linestyle = "dashed")
        plt.plot(self.agent0.x,self.agent0.y, label = "Agent_0")
        plt.plot(self.agent1.x,self.agent1.y, label = "Agent_1")
        plt.plot(self.agent2.x,self.agent2.y, label = "Agent_2")
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Trajectories of 3 Ships')
        # Add a legend
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.show()

    def close(self):
        pass

    def dist(self, x1, y1, x2, y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def reset(self, seed=None, options=None):
        self.epi += 1
        self.steps = 0
        self.theta = np.random.uniform(-np.pi,np.pi)
        self.length = 250
        self.xf = self.length*np.cos(self.theta)
        self.yf = self.length*np.sin(self.theta)

        observation = []
        self.agent0 = Agent(self.theta,self.xf,self.yf,0,0)
        observation.append(self.agent0.obs())
        self.agent1 = Agent(self.theta,self.xf,self.yf,0,0)
        observation.append(self.agent1.obs())
        self.agent2 = Agent(self.theta,self.xf,self.yf,0,0)
        observation.append(self.agent2.obs())

        observations = {}

        self.agents = self.possible_agents[:]
        for i in range(3):
            observations.update({f"agent_{i}":tuple(observation[i])})

        # print(observations)
        infos = {agent: {} for agent in self.agents}
        self.state = observations
        return observations, infos

    def step(self, actions):
        self.steps += 1
        observations = {}
        observation = []
        termination = {}
        terminate = []
        rewards = {}
        rew = []

        action = []
        for agent in self.agents:
            action.append(actions[agent])

        self.agent0.dynamics(action[0])
        observation.append(self.agent0.obs())
        terminate.append(self.agent0.done_func())
        rew.append(self.agent0.REWARD())

        self.agent1.dynamics(action[1])
        observation.append(self.agent1.obs())
        terminate.append(self.agent1.done_func())
        rew.append(self.agent1.REWARD())

        self.agent2.dynamics(action[2])
        observation.append(self.agent2.obs())
        terminate.append(self.agent2.done_func())
        rew.append(self.agent2.REWARD())


        for i in range(3):
            a = float(rew[i])
            observations.update({f"agent_{i}":tuple(observation[i])})
            termination.update({f"agent_{i}":terminate[i]})
            rewards.update({f"agent_{i}":a})

        # print('obs - ', observations)
        # print('rewards - ', rewards)
        # print('terminate - ',termination)
        # print('steps - ', self.steps)

        if self.steps > 360:
            termination = {agent: True for agent in self.agents}


        # if np.any(terminate) == True and self.epi % 10 == 0:
        #     self.render()

        infos = {agent: {} for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        
        return observations, rewards, termination, truncations, infos




# testing

# world = parallel_env()
# obs,info = world.reset()
# for i in range(400):
#     obs, rew, term, trun, info = world.step({'agent_0': 1,'agent_1': 0.75,'agent_2': 0.5})
# print(rew)
# world.render()