import gym
import multiagent_envs
import numpy as np
import time
from multiagent_envs.officeworld.ma_office import *
from multiagent_envs.taxi.multi_agent_taxi_planner import MultiAgentTaxiPlanner
from multiagent_envs.officeworld.multi_agent_officeworld_planner import MultiAgentOfficeWorldPlanner
from utils.RePReL_wrapper import RePReLWrapper
from utils.utils import create_gif


ma_env = gym.make('MultiAgentOffice-2A-medium-v0')
obs = ma_env.reset()
planner = MultiAgentOfficeWorldPlanner(obs, ma_env.target, target_binary=True)
current_time = int(time.time())
np.random.seed(current_time)
reprel_env = RePReLWrapper(ma_env, planner, use_heuristic=True)
obs = reprel_env.reset()
verbose=True
mode = "random"
agent_list = [1,2]
new_limit=0


for j in range(10000):
    
    if(mode=="input"):
        inp = input("Enter the steps for the agents")
        if(inp =="q"):
            break
        else:
            x,y = inp.strip().split(" ")

    elif(mode=="random"):
        x = np.random.randint(low=0,high=4)
        y = np.random.randint(low=0,high=4)

    action_list = [x,y]
    actions = {f"AGENT{agent_list[i]}": int(action_list[i])
                    for i in range(0, len(agent_list))} 
    
    print("Step Num ",j+1)
    print("Actions ")
    print(actions)
    obs, reward, done, info = reprel_env.step(actions)

    print("Plan")

    print(reprel_env.planner.plan)
    print(obs, reward, done, info, sep="\n")

    reprel_env.render()
    
    env_obs = ma_env.get_observation_from_state
    state_facts=np.array(env_obs[list(env_obs.keys())[0]][4*ma_env.n_agents:])

    target_facts = state_facts[:4]
    print(target_facts)
    
    flag=1
    if(verbose):
            print()
            count =  np.count_nonzero(target_facts)
            if(count>new_limit and flag==0):
                new_limit=count
                
                print(reprel_env.planner.plan)
                print()
                print()

            if(any(x > 1 for x in reward.values())):
                reprel_env.render()
                print(reprel_env.planner.plan)
                print()
                print()


    if(done["__all__"]):
        state_facts=np.array(obs)
        print(state_facts)
        break
