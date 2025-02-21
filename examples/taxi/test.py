
import gym
import multiagent_envs
import numpy as np
import time
from multiagent_envs.taxi.multi_agent_taxi_planner import MultiAgentTaxiPlanner
from utils.RePReL_wrapper import RePReLWrapper
from utils.utils import create_gif
from utils.task_extended_wrapper import TaskExtendedWrapper


ACTION_LOOKUP = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right',
    4: 'pick up',
    5: 'drop'
}

env = gym.make('MultiAgentTaxi-T2-P2-v0')
current_time = time.time()
obs = env.reset()
reset_time = time.time()
print("Reset Time: {}",reset_time-current_time)
np.random.seed(int(current_time))
mode = "random"
verbose=True
planner = MultiAgentTaxiPlanner(obs)
env = TaskExtendedWrapper(env, planner)
# env = RePReLWrapper(env, planner)
obs =  env.reset()
print(obs)


for i in range(200):
    print(env.observation_space)
    quit()

    if(mode=="input"):
        inp = input("Enter the steps for the agents")
        if(inp =="q"):
            break
        else:
            x,y = inp.strip().split(" ")
    
    elif(mode=="random"):
        x = np.random.randint(low=0,high=6)
        y = np.random.randint(low=0,high=6)
        
    

    
    x, y = int(x), int(y)
    
    obs, reward, done, info = env.step({"AGENT1":x, "AGENT2":y})
    t = time.time()
    print("Step {} time: {}".format(i+1,t-current_time))
    current_time = t
 
        
    # del info[list(info.keys())[0]]["passengers_picked"]
    # del info[list(info.keys())[0]]["passengers_dropped"]
    # del info[list(info.keys())[0]]["reward"]
    # info[list(info.keys())[0]]["actions"] = "\n{} | {}".format(ACTION_LOOKUP[x],ACTION_LOOKUP[y])

    # env.render(mode="save",saveloc="imgs/random", filename="random_{}".format(i),title="Random Policy 2", )

    if(verbose):
        # print(reprel_env.planner.plan)
        print(obs, reward, done, info, sep="\n")
        pass
    
    if(done["__all__"]):
        break

create_gif(save_name="Random RL3.gif",dir_name="imgs/random",save_dir="imgs",time_per_frame=4)