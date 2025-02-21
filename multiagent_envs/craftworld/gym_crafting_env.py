import numpy as np

from gym import Env, spaces
from gym.utils import seeding
import sys
from six import StringIO, b
import copy
import pygame
from gym import utils
from gym.envs.toy_text import discrete

from multiagent_envs.craftworld.crafting_base import CraftingBase

import re


ACTION_TEXT = ['Up',
            'Down',
            'Left',
            'Right',
           'Pickup',
           'Drop']

def add_space_bw(word):
    # Use regular expression to find words starting with capital letters
    # and add a space before them
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', word)


class GymCraftingEnv(CraftingBase, Env):
    """
    A Gym API to the crafting environment.
    """
    def __init__(self, **kwargs ):
        super().__init__(**kwargs)

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