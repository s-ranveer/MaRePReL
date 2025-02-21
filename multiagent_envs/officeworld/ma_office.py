# We would be implementing a custom officeworld environment
import copy
import os
import sys

import gym
# import gymnasium.spaces
import numpy as np
# # import pygame
# import glob
from ray.rllib import MultiAgentEnv
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.env import EnvContext
from ray.rllib.utils.typing import MultiAgentDict

from utils.utils import manhattan_distance
# We define the enumeration for the tasks
VISIT_A = 0
VISIT_B = 1
VISIT_C = 2
VISIT_D = 3
PICK_UP_MAIL = 4
PICK_UP_COFFEE = 5
VISIT_OFFICE = 6
DELIVER_MAIL = 7
DELIVER_COFFEE = 8


TARGET_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "M",
    5: "T",
    6: "O",
    7: "M",
    8: "C",
}

# We define the default reward/cost variables

DEFAULT_STEP_COST = -0.1
DEFAULT_INVALID_ACTION_COST = -1
DEFAULT_TERMINAL_REWARD = 100
DEFAULT_BUMP_COST = -10

# We define the possible actions for the agents
ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

INTEGER_TO_ACTIONS = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}
world_config = {
    "render_mode": False,
    "terminal_reward": 100,
    "step_cost": -0.1,
    "bump_mode": False,
    "tasks": [0,1,2,3],
    "layout":
        [["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
         ["X", " ", " ", " ", "X", " ", " ", " ", "X", " ", " ", " ", "X", " ", " ", " ", "X"],
         ["X", " ", "D", " ", " ", " ", "P", " ", " ", " ", "P", " ", " ", " ", "C", " ", "X"],
         ["X", " ", " ", " ", "X", "T", " ", " ", "X", " ", " ", " ", "X", " ", " ", " ", "X"],
         ["X", "X", " ", "X", "X", "X", " ", "X", "X", "X", " ", "X", "X", "X", " ", "X", "X"],
         ["X", " ", " ", " ", "X", " ", " ", " ", "X", " ", " ", " ", "X", " ", " ", " ", "X"],
         ["X", " ", "P", " ", " ", " ", "O", " ", " ", " ", "M", " ", " ", " ", "P", " ", "X"],
         ["X", " ", " ", " ", "X", " ", " ", " ", "X", " ", " ", " ", "X", " ", " ", " ", "X"],
         ["X", "X", " ", "X", "X", "X", " ", "X", "X", "X", " ", "X", "X", "X", " ", "X", "X"],
         ["X", " ", " ", " ", "X", " ", " ", " ", "X", " ", " ", "T", "X", " ", " ", " ", "X"],
         ["X", " ", "A", " ", " ", " ", "P", " ", " ", " ", "P", " ", " ", " ", "B", " ", "X"],
         ["X", " ", " ", " ", "X", "T", " ", " ", "X", " ", " ", " ", "X", " ", " ", " ", "X"],
         ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]]
}


def instantiate_world(config):
    """
    Method for instantiating a new world
    :param config:  The configuration for the world
    :return: The state of the world
    """
    state = dict()
    keys = list(config.keys())

    n_agents = config["n_agents"] if "n_agents" in keys else 2
    assert n_agents <= 4, "We can only have a maximum of 4 agents"

    
    state["layout"] = copy.deepcopy(config["layout"]) if "layout" in keys else world_config["layout"]
    state["agents"] = {f"AGENT{i + 1}": {} for i in range(n_agents)}
    state["goals"] = config["tasks"] if "tasks" in keys else world_config["tasks"]
    state["agents_bumped"] = config["agents_bumped"] if "agents_bumped" in keys else world_config["bump_mode"]
    state["agents_bumped_this_step"] = False
    state["tasks_completed"] = {task: None for task in state["goals"]}

    # For the visited locations, MO and TO represent the case when we visited the office
    # and had mail and coffee respectively in the inventory
    state["visited_locations"] = {
        "A": None,
        "B": None,
        "C": None,
        "D": None,
        "M": None,
        "T": None,
        "O": None,
        "MO": None,
        "TO": None
    }
    state["locations"] = {
        "A": [], "B": [], "C": [], "D": [], "M": [], "T": [], "O": []
    }
    empty_spaces = []

    if "randomize_task_locations" in keys and config["randomize_task_locations"]:
        task_count = {task: 0 for task in state["locations"].keys()}

        # We would replace the original locations in the layout with empty spaces
        for i in range(len(state["layout"])):
            for j in range(len(state["layout"][i])):
                if state["layout"][i][j] in state["locations"].keys():
                    task_count[state["layout"][i][j]] += 1
                    state["layout"][i][j] = " "
                    empty_spaces.append((i, j))

                if state["layout"][i][j] == " ":
                    empty_spaces.append((i, j))

       # Once, we have the empty spaces as well as the count, we would sample the new locations for the tasks
        for task, count in task_count.items():
            task_locations = np.random.choice(len(empty_spaces), size=count, replace=False)
            for location in task_locations:
                state["locations"][task].append(empty_spaces[location])
                # We would update the layout to have the tasks in the new locations
                state["layout"][empty_spaces[location][0]][empty_spaces[location][1]] = task
                # We would remove the location from the empty spaces
                empty_spaces.pop(location)

    else:
        # This is the case when we use the original map
        for i in range(len(state["layout"])):
            for j in range(len(state["layout"][i])):
                if state["layout"][i][j] in ["A", "B", "C", "D", "M", "T", "O"]:
                    state["locations"][state["layout"][i][j]].append((i, j))

        # We would get all the empty spaces in the environment
        empty_spaces = []
        for i in range(len(state["layout"])):
            for j in range(len(state["layout"][i])):
                if state["layout"][i][j] == " ":
                    empty_spaces.append((i, j))

    # We would sample without replacement, locations for the agents
    agent_locations = np.random.choice(len(empty_spaces), size=n_agents, replace=False)

    # We would assign the agents their current locations as well as their inventory
    for i, agent in enumerate(state["agents"].keys()):
        state["agents"][agent]["location"] = empty_spaces[agent_locations[i]]
        state["agents"][agent]["inventory"] = {"mail": False, "coffee": False}

    # The agent locations are in the format of (y, x)

    # We need to update the layout to have the agents in the environment as well as the objects
    for obj, locations in state["locations"].items():
        if locations:
            for location in locations:
                state["layout"][location[0]][location[1]] = obj

    # We would update the agent locations in the environment as well
    for agent, agent_info in state["agents"].items():
        state["layout"][agent_info["location"][0]][agent_info["location"][1]] = agent[5:]

    return state


def state_transitions(state, actions):
    """
    Method for transitioning the state of the world
    :param state: The current state
    :param actions: The actions for the different agents
    :return: The updated state for the world as well as the success of the actions for the agents
    """
    successful_action = {agent: False for agent in state["agents"].keys()}

    # We would randomly select the order of the agents
    agent_order = np.random.choice(list(state["agents"].keys()), size=len(state["agents"]), replace=False)

    # We would check if there was a bump in the current step
    bump_this_step = False
    state["agents_bumped_this_step"] = False
    for agent in agent_order:
        if(agent in actions.keys()):
            agent_action = actions[agent]
            # We would get the current location of the agent
            current_location = state["agents"][agent]["location"]
            new_location = (current_location[0] + ACTIONS[agent_action][0],
                            current_location[1] + ACTIONS[agent_action][1])

            # We would check if the new location is valid and move only when possible and location is valid and a
            # location that can be moved to
            object_at_agent_move_loc = ""
            if 0 <= new_location[0] < len(state["layout"]) and 0 <= new_location[1] < len(state["layout"][0]):
                # If there is another agent in the new location, we would not move
                for other_agent, other_agent_info in state["agents"].items():
                    if other_agent != agent and other_agent_info["location"] == new_location:
                        state["agents_bumped"] = True
                        state["agents_bumped_this_step"] = True
                        bump_this_step = True

                if state["layout"][new_location[0]][new_location[1]] != "X" \
                        and state["layout"][new_location[0]][new_location[1]] != "P" and not bump_this_step:
                    object_at_agent_move_loc = state["layout"][new_location[0]][new_location[1]]
                    state["layout"][current_location[0]][current_location[1]] = " "
                    state["agents"][agent]["location"] = new_location
                    state["layout"][new_location[0]][new_location[1]] = agent[5:]
                    successful_action[agent] = True

            # If the location we moved to was one of the visit locations, we would update the tasks completed
            if object_at_agent_move_loc in ["A", "B", "C", "D", "M", "T", "O"]:
                if state["visited_locations"][object_at_agent_move_loc] is None:
                    state["visited_locations"][object_at_agent_move_loc] = agent

                # We need to check if any of the goals have been completed

                # The Visit tasks
                if object_at_agent_move_loc == "A" and VISIT_A in state["goals"]:
                    state["tasks_completed"][VISIT_A] = agent

                if object_at_agent_move_loc == "B" and VISIT_B in state["goals"]:
                    state["tasks_completed"][VISIT_B] = agent

                if object_at_agent_move_loc == "C" and VISIT_C in state["goals"]:
                    state["tasks_completed"][VISIT_C] = agent

                if object_at_agent_move_loc == "D" and VISIT_D in state["goals"]:
                    state["tasks_completed"][VISIT_D] = agent

                # The pickup tasks
                if object_at_agent_move_loc == "M":
                    if PICK_UP_MAIL in state["goals"]:
                        state["tasks_completed"][PICK_UP_MAIL] = agent
                    state["agents"][agent]["inventory"]["mail"] = True

                if object_at_agent_move_loc == "T":
                    if PICK_UP_COFFEE in state["goals"]:
                        state["tasks_completed"][PICK_UP_COFFEE] = agent
                    state["agents"][agent]["inventory"]["coffee"] = True

                # When we visit the office, we have to check for the other goals as well
                if object_at_agent_move_loc == "O":
                    if VISIT_OFFICE in state["goals"]:
                        state["tasks_completed"][VISIT_OFFICE] = agent

                    # If the agent has visited the office, we would check if they have the mail and coffee and deliver
                    # them if they do
                    if state["agents"][agent]["inventory"]["mail"]:
                        if DELIVER_MAIL in state["goals"]:
                            state["tasks_completed"][DELIVER_MAIL] = agent
                        state["agents"][agent]["inventory"]["mail"] = False
                        if state["visited_locations"]["MO"] is None:
                            state["visited_locations"]["MO"] = agent

                    if state["agents"][agent]["inventory"]["coffee"]:
                        if DELIVER_COFFEE in state["goals"]:
                            state["tasks_completed"][DELIVER_COFFEE] = agent
                        state["agents"][agent]["inventory"]["coffee"] = False
                        if state["visited_locations"]["TO"] is None:
                            state["visited_locations"]["TO"] = agent

    # We need to make sure that the objects are back in place once the agent has moved from the location
    for obj, locations in state["locations"].items():
        if locations:
            for location in locations:
                if state["layout"][location[0]][location[1]] == " ":
                    state["layout"][location[0]][location[1]] = obj

    return state, successful_action


def get_rewards(state, successful_actions, r_variables, bump_mode=False):
    """
    Method for getting the rewards for the agents
    :param successful_actions: If the actions are successful
    :param config: The current configuration
    :param state: The current environment state
    :return: The rewards for the agents
    """


    rewards = {agent: r_variables["step_cost"] for agent in state["agents"].keys()}

    # If all the tasks are completed, we would give the terminal reward to the agents
    if bump_mode and state["agents_bumped_this_step"]:
        rewards = {agent: rewards[agent]  + r_variables["bump_cost"] for agent in state["agents"].keys()}
    
    if all(state["tasks_completed"].values()):
        rewards = {agent: rewards[agent] + r_variables["terminal_reward"] for agent in state["agents"].keys()}

    if not all(successful_actions.values()):
        for agent in state["agents"].keys():
            if not successful_actions[agent]:
                rewards[agent] += r_variables["invalid_action_cost"]

    return rewards


# We would implement a ray multi-agent environment for the officeworld
class MultiAgentOfficeWorld(MultiAgentEnv):
    def __init__(self, t_config: EnvContext):

        super().__init__()

        self.random_seed = None
        self.with_state = t_config.get("with_state", False)
        self.config = t_config
        self.state = instantiate_world(self.config)
        self.config_keys = list(t_config.keys())
        self.n_agents = self.config.get("n_agents", 2)
        self.reward_variables = {"step_cost":self.config["step_cost"] if "step_cost" in self.config_keys else DEFAULT_STEP_COST,
                                 "terminal_reward": self.config["terminal_reward"] if "terminal_reward" in self.config_keys else DEFAULT_TERMINAL_REWARD,
                                 "invalid_action_cost": self.config["invalid_action_cost"] if "invalid_action_cost" in self.config_keys else DEFAULT_INVALID_ACTION_COST,
                                 "bump_cost": self.config["bump_cost"] if "bump_cost" in self.config_keys else DEFAULT_BUMP_COST}
        self.horizon = self.config.get("horizon",200)
        self.grid_config = np.array(t_config.get("layout", world_config["layout"]))
        self.render_mode = self.config.get("render_mode",False)
        self.agents = list(self.state["agents"].keys())
        self.step_results = {}
        self.episode_length = 0
        self.episode_reward = 0
        self.user_data_fields = []
        self.target_diagnostics = ["episode_reward", "episode_length", "success",
                                   "tasks_completed", "total_tasks"]
        self.target = self.config.get("tasks", world_config["tasks"])
        self.user_data_fields = self.target_diagnostics

        if any(x in [VISIT_A, VISIT_B, VISIT_C, VISIT_D, VISIT_OFFICE] for x in self.state["goals"]):
            self.target_diagnostics.append("Visits_Completed")
            self.target_diagnostics.append("Visit_Tasks")
          

        if any(x in [PICK_UP_MAIL, PICK_UP_COFFEE] for x in self.state["goals"]):
            self.target_diagnostics.append("Pickups_Completed")
            self.target_diagnostics.append("Pickup_Tasks")
          
        if any(x in [DELIVER_MAIL, DELIVER_COFFEE] for x in self.state["goals"]):
            self.target_diagnostics.append("Delivers_Completed")
            self.target_diagnostics.append("Deliver_Tasks")
         

       
        self.observation_size = 4*len(self.agents) + 9
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.observation_size,), dtype=np.float32)
        if self.with_state:
            self.observation_space = gym.spaces.Dict({"obs":gym.spaces.Box(-1, 1, shape=tuple([self.observation_size]), dtype='float32'), ENV_STATE: gym.spaces.Box(-1, 1, shape=tuple([self.observation_size]), dtype='float32')})   
        self.observation_space_dict = {agent: self.observation_space for agent in self.agents}
        self.nA = len(ACTIONS)
        self.action_space = gym.spaces.Discrete(self.nA)
        self.action_space_dict = {agent: self.action_space for agent in self.agents}
        self._agent_ids = self.agents

        if(self.render_mode):
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((800,800))   
            pygame.display.set_caption('Office World')


    @property
    def get_agent_ids(self):
        return list(self._agent_ids)
        
    @property
    def get_observation_from_state(self):
        """
        Method for getting the observation from the state
        :param state: The current environment state
        :return: The observation for the different agents
        """
        state=self.state
        # We would get the observation for the different agents
        observations = {agent: [] for agent in state["agents"].keys()}

        for agent, agent_info in state["agents"].items():
            # We would append the location for the agent
            observations[agent].append(agent_info["location"][1] / 100)
            observations[agent].append(agent_info["location"][0] / 100)

            # We would append the location for the other agents
            for other_agent, other_agent_info in state["agents"].items():
                if other_agent != agent:
                    observations[agent].append(other_agent_info["location"][1] / 100)
                    observations[agent].append(other_agent_info["location"][0] / 100)

            # We would append the inventory for the agent
            observations[agent].append(int(agent_info["inventory"]["mail"]))
            observations[agent].append(int(agent_info["inventory"]["coffee"]))

            # We would append the inventory for the other agents
            for other_agent, other_agent_info in state["agents"].items():
                if other_agent != agent:
                    observations[agent].append(int(other_agent_info["inventory"]["mail"]))
                    observations[agent].append(int(other_agent_info["inventory"]["coffee"]))

            # We would append the remaining details about the visited
            if state["visited_locations"]["A"] is not None:
                if(state["visited_locations"]["A"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["A"][5:]) / 10)
            else:
                observations[agent].append(0)

            if state["visited_locations"]["B"] is not None:
                if(state["visited_locations"]["B"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["B"][5:]) / 10)
            else:
                observations[agent].append(0)

            if state["visited_locations"]["C"] is not None:
                if(state["visited_locations"]["C"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["C"][5:]) / 10)
            else:
                observations[agent].append(0)

            if state["visited_locations"]["D"] is not None:
                if(state["visited_locations"]["D"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["D"][5:]) / 10)
            else:
                observations[agent].append(0)

            if state["visited_locations"]["M"] is not None:
                if(state["visited_locations"]["M"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["M"][5:]) / 10)
            else:
                observations[agent].append(0)

            if state["visited_locations"]["T"] is not None:
                if(state["visited_locations"]["T"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["T"][5:]) / 10)
            else:
                observations[agent].append(0)

            if state["visited_locations"]["O"] is not None:
                if(state["visited_locations"]["O"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["O"][5:]) / 10)
            else:
                observations[agent].append(0)

            if state["visited_locations"]["MO"] is not None:
                if(state["visited_locations"]["MO"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["MO"][5:]) / 10)
            else:
                observations[agent].append(0)

            if state["visited_locations"]["TO"] is not None:
                if(state["visited_locations"]["TO"]==agent): 
                    observations[agent].append(1.0)
                else:
                    observations[agent].append(int(state["visited_locations"]["TO"][5:]) / 10)
            else:
                observations[agent].append(0)

        env_state = []
        for i, e in enumerate(observations[list(observations.keys())[0]]):
            if(i>7 and e>0):
                env_state.append(1)
            else:
                env_state.append(e)

        # print("Env_State: ",env_state)

        # We would convert the observations to a numpy array
        for agent, agent_info in state["agents"].items():
            if self.with_state:
                
                qmix_obs = observations[agent]
                
                
                
                observations[agent] = {"obs":np.array(qmix_obs, dtype=float), ENV_STATE:np.array(env_state)}
            
            else:
                observations[agent] = np.array(observations[agent], dtype=float)


        # We would return the observations
        return observations


    def set_seed(self,seed, verbose=False):
        if(verbose):
            print("Seed Set to : ",seed)
        self.random_seed= seed


    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        actions = {}
        agents = self.get_agent_ids

        for a in agents:
            actions[a] = np.random.choice(self.nA)

        return actions

    def reset(self, **kwargs):
        np.random.seed(self.random_seed)
        self.episode_reward = 0
        self.state = instantiate_world(self.config)
        self.step_results = {}
        self.episode_length = 0
        self.agents = list(self.state["agents"].keys())
        return self.get_observation_from_state

    def step(self, action_dict):
        # Convert the numerical actions to the actual actions for the agents
        action_dict = {agent: INTEGER_TO_ACTIONS[action] for agent, action in action_dict.items()}

        # Perform the state transitions and increment the step count
        self.state, step_success = state_transitions(self.state, action_dict)
        self.episode_length += 1

        # Get the new observations, rewards, infos, and dones
        observations = self.get_observation_from_state
        rewards = get_rewards(self.state, step_success, self.reward_variables)
        dones = {"__all__": all(self.state["tasks_completed"].values()) or self.episode_length >= self.horizon}

        self.episode_reward = self.episode_reward + sum(rewards.values())

        for agent in self.agents:
            dones[agent] = dones["__all__"]

        # Get the info
        infos = {agent: {"bumped": self.state["agents_bumped"],
                         "episode_length": self.episode_length} for agent in self.agents}

        self.step_results = {"state": self.state, "observations": observations, "rewards": rewards,
                             "dones": dones, "infos": infos}
        return observations, rewards, dones, infos
    

    def render(self, mode="text"):
        for i in range(len(self.state["layout"])):
            for j in range(len(self.state["layout"][i])):
                print(self.state["layout"][i][j], end=" ")
            print()


    def get_diagnostics(self):
        
        diagnostics = {target: 0 for target in self.target_diagnostics}
        diagnostics["total_tasks"] = len(self.state["tasks_completed"])
        for task, _ in self.state["tasks_completed"].items():
            if "Visit_Tasks" in diagnostics and task in [VISIT_A, VISIT_B, VISIT_C, VISIT_D, VISIT_OFFICE]:
                    diagnostics["Visit_Tasks"] += 1
            if "Pickup_Tasks" in diagnostics and task in [PICK_UP_MAIL, PICK_UP_COFFEE]:
                    diagnostics["Pickup_Tasks"] += 1
            if "Deliver_Tasks" in diagnostics and task in [DELIVER_MAIL, DELIVER_COFFEE]:
                    diagnostics["Deliver_Tasks"] += 1  

        # We would get the total episode return
        total_return = sum([reward for reward in self.step_results["rewards"].values()])
        diagnostics["episode_reward"] = self.episode_reward

        # We would get the episode length
        diagnostics["episode_length"] = self.episode_length

        # We would get the overall success which happens when all the tasks are completed
        diagnostics["success"] = 1 if all(self.state["tasks_completed"].values()) else 0

        # We would get the number of tasks completed
        diagnostics["tasks_completed"] = sum([1 for task in self.state["tasks_completed"].values() if task is not None])

        # We would get the total number of visit tasks
        for task, completion_agent in self.state["tasks_completed"].items():
            if task in [VISIT_A, VISIT_B, VISIT_C, VISIT_D, VISIT_OFFICE] and completion_agent is not None and "Visit_Tasks" in diagnostics:
                    diagnostics["Visits_Completed"] += 1
                
        # We would get the total number of pick up tasks
            if task in [PICK_UP_MAIL, PICK_UP_COFFEE] and completion_agent is not None and "Pickup_Tasks" in diagnostics:
                    diagnostics["Pickups_Completed"] += 1

        # We would get the total number of deliver tasks
            if task in [DELIVER_MAIL, DELIVER_COFFEE] and completion_agent is not None and "Deliver_Tasks" in diagnostics:
                    diagnostics["Delivers_Completed"] += 1



        return diagnostics


    def calculate_heuristic(self, from_loc, target):
        
        target= target
        target_x, target_y = np.where(self.grid_config==TARGET_MAP[target])

        if from_loc in self._agent_ids:
            agent = from_loc
            obs = self.get_observation_from_state
            obs = obs[agent]
            return manhattan_distance((obs[0]*100, obs[1]*100), (target_x[0], target_y[0]))
        
        else:
            if isinstance(from_loc, int):
                from_x, from_y = np.where(self.grid_config==TARGET_MAP[from_loc])
                return manhattan_distance((from_x[0],from_y[0]), (target_x[0], target_y[0]))

if __name__ == "__main__":
    # We would test the environment
    config = world_config
    env = MultiAgentOfficeWorld(config)
    obs = env.reset()
    env.render()
    d = {"__all__": False}
    while not d["__all__"]:
        agent_actions = env.action_space_sample()
        obs, r, d, i = env.step(agent_actions)

        env.render()
        print(env.state["goals"])
        
        print(obs)
        print(f"{env._agent_ids[0]}: {INTEGER_TO_ACTIONS[agent_actions[env._agent_ids[0]]]}, \t  {env._agent_ids[1]}: {INTEGER_TO_ACTIONS[agent_actions[env._agent_ids[1]]]}")
        print(r)
        print(d)
        print(i)
        print(env.get_diagnostics())
        if d["__all__"]:
            break