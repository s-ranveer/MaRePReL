
#Ray IMPORTS
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from ray.rllib.env.env_context import EnvContext
from utils.utils import *
import logging
import pygame
from random import Random
from typing import Iterable, FrozenSet, List, Set, Union, Tuple, Dict
from gym.spaces import MultiDiscrete
import numpy as np
from gym.utils import seeding
import sys
from six import StringIO, b
import copy
import pygame
from gym import utils
from gym.envs.toy_text import discrete
from multiagent_envs.craftworld.gym_crafting_env import GymCraftingEnv
import re


OBJECTS = ['rock', 'hammer', 'tree', 'axe', 'bread', 'sticks', 'house', 'wheat']

class MACraftingEnv(GymCraftingEnv, MultiAgentEnv):
    """
    A Gym API to the crafting environment.
    """
    def __init__(self,  n_agents=2, agent_ids=None, **kwargs ):
        super().__init__(**kwargs)

        """
        Inherits the following: 
        size=[10,10]: The size of the grid before rendering. [10,10] indicates that 
            there are 100 positions the agent can have.
        res=39: The resolution of the observation, must be a multiple of 3.
        add_objects=[]: List of objects that must be present at each reset.
        visible_agent=True: Whether to render the agent in the observation.
        state_obs=False: If true, the observation space will be the positions of the
            objects, agents, etc.
        few_obj=False: If true, only one of each object type will be spawned per episode
        use_exit=False: If true, an additional action will allow the agent to choose when
            to end the episode
        success_function=None: A function applied to two states which evaluates whether 
            success was obtained between those states.
        pretty_renderable=True: If True, env will load assets for pretty rendering.
        fixed_init_state=False: If True, env will always reset to the same state, stored in 
            self.FIXED_INIT_STATE
        """

        #Minimum 2 agents for Multiagent
        assert n_agents>=2 , "The number of agents should atleast be 2"
        self.n_agents = n_agents
        self.agent_ids = []
        
        #Default Agent Ids are AGENT1 and AGENT2
        if agent_ids is None:
            for i in range(self.n_agents):
                agent = "AGENT" + str(i + 1)
                self.agent_ids.append(agent)
            
        else:
            assert isinstance(agent_ids,dict), "The argument agent_id should be either a dictionary or None"
            self.agent_ids = agent_ids


        self.action_space = MultiDiscrete([self.nA]*n_agents)
        if self.state_obs:
            assert (self.few_obj)
            self.state_space_size = len(OBJECTS) * 2 * self.max_num_per_obj + (2 + 1 + 1)*n_agents
            self.observation_space = MultiDiscrete([2]*self.state_space_size)
            self.state_space = self.observation_space
            self.goal_space = self.observation_space
        elif self.state_fusion:
            assert (self.few_obj)
            self.max_num_per_obj = 3
            self.image_height = self.nrow
            self.image_width = self.ncol
            self.image_channels = 1
            # the agent pos, objects pos , holding flag, hunger flag
            self.state_space_size = self.grid_dim + len(OBJECTS) * 2 * self.max_num_per_obj
            self.object_vector_size = len(OBJECTS) * 2 * self.max_num_per_obj
            if self.visible_neighbour:
                self.state_space_size += 4
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_space_size,))
        else:
            self.observation_space = spaces.Box(low=0, high=1., shape=((self.nrow + 1) * self.res * self.ncol * self.res * 3,))

    def reset(self, **kwargs):
        self._reset_from_init(**kwargs)
        obs = self.get_env_obs()
        self.init_obs = obs.copy()
        obs = self.get_obs()
        return obs
        
    def step(self, a):
        r,d, info = self._step_internal(a)
        obs = self.get_obs()
        return obs,r,d,info
    
    def sample_action(self):
        a = self.action_space.sample()
        return a
        
    def action_meaning(self, a):
        return ACTION_TEXT[a]

    def render(self, mode="rgb", wait_time=200):
        
        cworld = self.pretty_render(mode=mode)
        
        task = add_space_bw(self.tasks[self.task_id])
        rem =f"Task: {task}"
        last_action = f"Last Action: {self.action_meaning(self.lastaction)}"
        if self.state["hunger"]==1.0:
            h = "hungry" 
        else:
            h = "not hungry"
        hungry = f"Character is {h}"
        # Create the display
        self.surf.fill((0, 0, 0))   
        self.surf.blit(cworld, (0,0))

        if(self.render_with_info):
            size=self.screen_size
            title = self.add_text("Craft World", "title")
            gs = self.add_text("Game State", "body")
            agent_loc = self.add_text(f"Agent at {self.state['agent']}","body")
            if(self.state['holding']==""):
                agent_holding = self.add_text(f"Agent is holding nothing","body")
            else:
                agent_holding = self.add_text(f"Agent is holding {self.state['holding']}","body")
            tasks_rem = self.add_text(rem, "body")
            hunger = self.add_text(hungry, "body")

            ei = self.add_text("Episode Stats", "body")
            step_c = self.add_text(f"Step Count: {self.state['count']}", "body")
            last_rew = self.add_text(f"Last Reward {self.lastreward}","body")
            ep_rew = self.add_text(f"Episode Reward: {self.episode_reward}", "body")
            last_a = self.add_text(last_action, "body")

            oc = self.add_text("Object Counts", "body")
            rock = self.add_text(f"Rock: {self.state['object_counts']['rock']}", "body")
            hammer = self.add_text(f"Hammer: {self.state['object_counts']['hammer']}", "body")
            tree = self.add_text(f"Tree: {self.state['object_counts']['tree']}", "body")
            axe = self.add_text(f"Axe: {self.state['object_counts']['axe']}", "body")
            bread = self.add_text(f"Bread: {self.state['object_counts']['bread']}", "body")
            sticks = self.add_text(f"Stick: {self.state['object_counts']['sticks']}", "body")
            house = self.add_text(f"House: {self.state['object_counts']['house']}", "body")
            wheat = self.add_text(f"Wheat: {self.state['object_counts']['wheat']}", "body")
            
            self.surf.blit(title, (int(size*0.03), int(size*1.005)))
            self.surf.blit(gs, (size*0.05, size*1.05) )
            self.surf.blit(agent_loc, (size*0.06, size*1.08))
            self.surf.blit(agent_holding, (size*0.06, size*1.11))
            self.surf.blit(hunger,(size*0.06, size*1.14))
            self.surf.blit(tasks_rem, (size*0.06, size*1.17))

            self.surf.blit(ei,(size*0.33, size*1.05) )
            self.surf.blit(step_c,(size*0.34, size*1.08) )
            self.surf.blit(last_a,(size*0.34, size*1.11) )
            self.surf.blit(last_rew,(size*0.34, size*1.14) )
            self.surf.blit(ep_rew,(size*0.34, size*1.17) )

            self.surf.blit(oc,(size*0.63, size*1.05) )
            self.surf.blit(rock,(size*0.64, size*1.08) )
            self.surf.blit(hammer,(size*0.64, size*1.11) )
            self.surf.blit(tree,(size*0.64, size*1.14) )
            self.surf.blit(axe,(size*0.64, size*1.17) )

            self.surf.blit(bread,(size*0.77, size*1.08) )
            self.surf.blit(sticks,(size*0.77, size*1.11) )
            self.surf.blit(house,(size*0.77, size*1.14) )
            self.surf.blit(wheat,(size*0.77, size*1.17) )

            
        pygame.display.flip()
        pygame.time.wait(wait_time) 


if __name__ == "__main__":
    env = GymCraftingEnv(task_id=1, screen_size=800, render_with_info=True)
    
    obs = env.reset()
    
    i=0
    done = False
    print(obs)
    while i<=1000:
        if(done):
            break
        a = env.sample_action()
        obs, rew, done, info = env.step(a)
    
        env.render(mode="human")
        print(f"Step: {i+1}\nObs: {obs}\nRew: {rew}\nDone: {done}\n")
        i = i+1