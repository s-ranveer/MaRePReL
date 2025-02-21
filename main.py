import gym
import multiagent_envs
import time
import argparse
import json
from multiagent_envs.taxi.ma_taxi import MA_TaxiWorld
from multiagent_envs.taxi.taxiworld_gen import *
from multiagent_envs.taxi.ma_taxi import DEFAULT_CONFIG
parser = argparse.ArgumentParser()
parser.add_argument(
    "--iters",
    type=int,
    default=200,
    help="Number of iterations to train.")

parser.add_argument(
    "--env",
    type=str,
    default="MA-Taxi-v0",
    help="Environment")

parser.add_argument(
    "--ckpt-freq",
    type=int,
    default=100,
    help="Checkpoint frequency")

parser.add_argument(
    "--ckpt-num",
    type=int,
    default=10,
    help="Checkpoint frequency")

parser.add_argument(
    "--ckpt-path",
    type=str,
    default=None,
    help="Checkpoint path")

parser.add_argument(
    "--render",
    action='store_true',
    default=False,
    help="Render environment")

parser.add_argument(
    "--train-batch-size",
    type=int,
    default=20,
    help="Training Batch Size")


parser.add_argument(
    "--evaluation-steps",
    type=int,
    default=100,
    help="Number of steps for evaluation")

DEFAULT_CONFIG["random_seed"] = 42
parser.add_argument(
        "--env-config",
        default={
            't_config':DEFAULT_CONFIG
                },
        type= json.loads,
        help="Environment configurations.",
    )

if __name__ == "__main__":
    args = parser.parse_args()
    ENV_CONFIG = args.env_config.get("t_config")
    env = MA_TaxiWorld(ENV_CONFIG)
    choice="normal"
    obs = env.reset()

    if(choice=="normal"):
        
        # print(obs)
        env.render()
        o, r, d, i = env.step({'AGENT1':2, 'AGENT2':0})
        env.render()
        time.sleep(0.3)

        o, r, d, i = env.step({'AGENT1':2, 'AGENT2':0})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':2, 'AGENT2':2})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':2, 'AGENT2':2})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':2, 'AGENT2':4})
        env.render()
        print(i)
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':2, 'AGENT2':1})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':1, 'AGENT2':1})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':1, 'AGENT2':1})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':1, 'AGENT2':1})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':4, 'AGENT2':3})
        print(i)
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':0, 'AGENT2':1})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':0, 'AGENT2':1})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':0, 'AGENT2':1})
        env.render()
        time.sleep(0.3)
        o, r, d, i = env.step({'AGENT1':5, 'AGENT2':1})
        env.render()
        time.sleep(0.3)
        print(i)
        o, r, d, i = env.step({'AGENT1':1, 'AGENT2':5})
        env.render()
        time.sleep(0.3)
        print(i)
        print(d)
        # print(obs)

    else:
        from multiagent_envs.taxi.multi_agent_taxi_planner import MultiAgentTaxiPlanner


        obs = env.reset()
        obs = dict()
        # # #
        # # # # # The observation in case when we have both picked and dropped the passenger
        obs_pickup_drop = np.array([0.1, 0.8, 0.9, 0.7,
                                    0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                                    0. , 0. , 0. , 0. , 0.1 , 0. , 0. , 1. , 0. ,
                                    0. , 0. , 0. , 0. , 0.2 , 0. , 0. , 0. , 1. ,
                                    1. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. ])
        obs["Agent1"] = obs_pickup_drop
        obs["Agent2"] = obs_pickup_drop

        # # The observation in case when we have only picked the passenger
        # obs_pickup_only = np.array([0.3, 0.9, 0.6, 0.5,
        #                             0., 0., 0., 0., 0.2, 1., 0., 0., 0.,
        #                             0., 0., 1., 0., 0.0, 0., 1., 0., 0.,
        #                             0., 0., 0., 0., 0.1, 1., 0., 0., 0.,
        #                             1., 0., 0., 0., 0., 0., 0., 0., 1.])
        #
        # obs["Agent1"] = obs_pickup_only
        #
        # obs["Agent2"] = obs_pickup_only

        planner = MultiAgentTaxiPlanner(obs)
        plan = planner.get_plan(obs)
        planner.plan = plan
        abstract_obs = planner.get_abstract_obs(obs, planner.next_tasks(obs))
        print(planner.is_task_done(obs, planner.next_tasks(obs)))
        # print(abstract_obs)
        # env.render()
        # o, r, d, i = env.step({'AGENT1_pickup': 1, 'AGENT2': 2})
        # env.render()
        # time.sleep(0.3)
        #
        # o, r, d, i = env.step({'AGENT1': 1, 'AGENT2': 1})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1': 1, 'AGENT2': 3})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1': 0, 'AGENT2': 2})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1': 3, 'AGENT2': 2})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1': 2, 'AGENT2': 3})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1':0, 'AGENT2':3})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1':0, 'AGENT2':1})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1':0, 'AGENT2':1})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1':0, 'AGENT2':1})
        # env.render()
        # time.sleep(0.3)
        # o, r, d, i = env.step({'AGENT1':5, 'AGENT2':1})
        # env.render()
        # time.sleep(0.3)
        # print(i)
        # o, r, d, i = env.step({'AGENT1':1, 'AGENT2':5})
        # env.render()
        # time.sleep(0.3)
        # print(i)
        # print(d)
        # print(obs)
