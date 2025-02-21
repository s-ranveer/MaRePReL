import random
import gym
import copy
import numpy as np
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune import register_env

from multiagent_envs.collect_and_build.env_descriptors import *
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env import EnvContext


world_config = {
    "n_agents": 2,
    "resource_tiles": [wood_medium, wood_medium, stone_small, stone_medium],
    "horizon": 10000,
    "step_cost": -0.1,
    "terminal_reward": 100,
    "to_build": [house, mansion],
    "terrain": [["X", "X", " ", " ", " ", " ", " ", "X", "X"],
                ["X", " ", " ", "X", " ", "X", " ", " ", "X"],
                [" ", " ", "X", " ", " ", " ", "X", " ", " "],
                [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                [" ", " ", "X", " ", " ", " ", "X", " ", " "],
                ["X", " ", " ", "X", " ", "X", " ", " ", "X"],
                ["X", "X", " ", " ", " ", " ", " ", "X", "X"]]
}

ACTIONS = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "COLLECT",
    5: "BUILD"
}

ORIENTATION = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3
}


# We would now define how to instantiate the environment using the above configuration
def instantiate_world(config):
    world_state = {}
    # We would first initialize the layout of the world
    world_state["layout"] = copy.deepcopy(config["terrain"])
    
    # Possible spawn points for the agents, buildings and resource tiles (Need to be empty spaces in the layout)
    spawn_points = [(i, j) for i in range(len(config["terrain"])) for j in range(len(config["terrain"][0]))
                    if config["terrain"][i][j] == " "]

    # We would assert that the number of resource tiles is exactly equal to the resource tile count
    assert len(config["resource_tiles"]) == RESOURCE_TILES_COUNT

    # We may want to assert that the resource total equals the resources required to construct all the buildings
    # Otherwise, we may go into states where the buildings cannot be built
    # assert (sum([tile["count"] for tile in config["resource_tiles"]]) ==
    #         sum([building["cost"]["wood"] + building["cost"]["stone"] for building in config["to_build"]]))

    # We would get the unique resources
    unique_resources = list(set([tile["type"] for tile in config["resource_tiles"]]))

    random.shuffle(spawn_points)

    # We would now spawn the resource tiles
    world_state["resource_tiles"] = {}
    for i, tile in enumerate(config["resource_tiles"]):
        loc = spawn_points.pop()
        world_state["resource_tiles"][i] = {
            "type": tile["type"],
            "count": tile["count"],
            "location": loc
        }
        world_state["layout"][loc[0]][loc[1]] = type_to_layout[tile["type"]]
    
    # We would spawn the buildings
    world_state["buildings"] = {}
    for i, building in enumerate(config["to_build"]):
        loc = spawn_points.pop()
        world_state["buildings"][i] = {
            "type": building["type"],
            "location": loc,
            "resources_needed": building["cost"],
            "resources_used_to_build_so_far": {resource: 0 for resource in building["cost"].keys()},
            "built": False
        }
        world_state["layout"][loc[0]][loc[1]] = type_to_layout[building["type"]]
    
    # We would now spawn the agents
    world_state["agents"] = {}
    for i in range(config["n_agents"]):
        loc = spawn_points.pop()
        agent_id = i+1
        world_state["agents"][f"AGENT{i+1}"] = {
            "id": agent_id,
            "location": spawn_points.pop(),
            "orientation": "UP",
            "inventory": {resource: 0 for resource in unique_resources},
        }
        world_state["layout"][loc[0]][loc[1]] = str(agent_id)

    # We would return the world state
    return world_state

def set_layout(state, config):
    """
    This function would set the layout of the world based on the current state

    """
    layout = copy.deepcopy(config["terrain"])

    # We would now set the resource tiles for the layout
    for tile in state["resource_tiles"].values():
        if tile["count"] > 0:
            layout[tile["location"][0]][tile["location"][1]] = type_to_layout[tile["type"]]
    
    # We would now set the layout for the buildings
    for building in state["buildings"].values():
        layout[building["location"][0]][building["location"][1]] = type_to_layout[building["type"]].lower() if building["built"] \
                                                                                            else type_to_layout[building["type"]]
    
    # We would now set the layout for the agents
    for agent in state["agents"].values():
        layout[agent["location"][0]][agent["location"][1]] = str(agent["id"])
    
    state["layout"] = layout
    
    return state

# We would now define the function to move the agents
def move_agent(state, agent, direction):
    # We would get the current location of the agent
    current_loc = state["agents"][agent]["location"]
    if direction == "UP":
        new_loc = (current_loc[0] - 1, current_loc[1])
    elif direction == "DOWN":
        new_loc = (current_loc[0] + 1, current_loc[1])
    elif direction == "LEFT":
        new_loc = (current_loc[0], current_loc[1] - 1)
    elif direction == "RIGHT":
        new_loc = (current_loc[0], current_loc[1] + 1)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # We would set the orientation of the agent (Orientation changes regardless of if the action is successful or not)
    orientation_changed = False

    if state["agents"][agent]["orientation"] != direction:
        state["agents"][agent]["orientation"] = direction
        orientation_changed = True

    # We would not allow the agent to move out of the grid
    if new_loc[0] < 0 or new_loc[0] >= len(state["layout"]) or new_loc[1] < 0 or new_loc[1] >= len(state["layout"][0]):
        return state, False or orientation_changed
    
    # We would not allow the agent to move to a location that is already occupied
    if state["layout"][new_loc[0]][new_loc[1]] != " ":
        return state, False or orientation_changed

    # We would now update the location of the agent
    state["agents"][agent]["location"] = new_loc

    # We would now update the layout of the world
    state["layout"][current_loc[0]][current_loc[1]] = " "
    state["layout"][new_loc[0]][new_loc[1]] = str(state["agents"][agent]["id"])

    # We would return the updated state and a flag indicating if the action was successful
    return state, True
    
# We would now define the function to collect resources
def collect_resource(state, agent):

    # We would get the current location of the agent
    current_loc = state["agents"][agent]["location"]

    # We would get resource tile at the location facing the agent (i.e the location in the direction of the agent's orientation)
    if state["agents"][agent]["orientation"] == "UP":
        loc = (current_loc[0] - 1, current_loc[1])
    elif state["agents"][agent]["orientation"] == "DOWN":
        loc = (current_loc[0] + 1, current_loc[1])
    elif state["agents"][agent]["orientation"] == "LEFT":
        loc = (current_loc[0], current_loc[1] - 1)
    elif state["agents"][agent]["orientation"] == "RIGHT":
        loc = (current_loc[0], current_loc[1] + 1)

    # If the agent inventory is full, we would return the state and a flag indicating the action was not successful
    agent_inventory = state["agents"][agent]["inventory"]

    # If the location is out of the grid, we would return the state and a flag indicating the action was not successful
    if loc[0] < 0 or loc[0] >= len(state["layout"]) or loc[1] < 0 or loc[1] >= len(state["layout"][0]):
        return state, False

    # If the location does not have a resource tile, we would return the state and a flag indicating the action was not successful
    if state["layout"][loc[0]][loc[1]] not in type_to_layout.values():
        return state, False

    # We would iterate over the different resource tiles to find the one at the location
    resource_collected = None
    for tile_id, tile in state["resource_tiles"].items():
        if tile["location"] == loc and agent_inventory[tile["type"]] < MAX_INVENTORY_PER_RESOURCE:
            # If we have a resource tile at the location, we would collect the resource,
            # update the agent's inventory and the resource tile count
            state["agents"][agent]["inventory"][tile["type"]] += 1
            state["resource_tiles"][tile_id]["count"] -= 1
            resource_collected = (tile_id, tile["type"])
            break
    
    # If a resource was collected, the action was successful
    if resource_collected:
        # If the resource tile count is 0, we would remove the resource tile from the layout
        if state["resource_tiles"][resource_collected[0]]["count"] == 0:
            state["layout"][loc[0]][loc[1]] = " "

        # We would return the updated state and a flag indicating the action was successful
        return state, True
    
    # This point would only be reached if the resource tile was not found at the location and the action was not successful
    return state, False



# We would now define the function to build
def build(state, agent):
    # We would get the current location of the agent
    current_loc = state["agents"][agent]["location"]

    # We would get the building at the location facing the agent
    # (i.e the location in the direction of the agent's orientation)
    if state["agents"][agent]["orientation"] == "UP":
        loc = (current_loc[0] - 1, current_loc[1])
    elif state["agents"][agent]["orientation"] == "DOWN":
        loc = (current_loc[0] + 1, current_loc[1])
    elif state["agents"][agent]["orientation"] == "LEFT":
        loc = (current_loc[0], current_loc[1] - 1)
    elif state["agents"][agent]["orientation"] == "RIGHT":
        loc = (current_loc[0], current_loc[1] + 1)
    else:
        raise ValueError(f"Invalid orientation: {state['agents'][agent]['orientation']}")
    
    # If the location is out of the grid, we would return the state and a flag indicating the action was not successful
    if loc[0] < 0 or loc[0] >= len(state["layout"]) or loc[1] < 0 or loc[1] >= len(state["layout"][0]):
        return state, False
    
    # If the location does not have a building, we would return the state and a flag indicating the action was not successful
    if state["layout"][loc[0]][loc[1]] not in type_to_layout.values():
        return state, False
    
    # We would iterate over the different buildings to find the one at the location
    worked_on_building = None
    for building_id, building in state["buildings"].items():
        if building["location"] == loc:
            # If the agent has some resource that is needed to build the building, we would use the resource to build the building
            resources_needed = building["resources_needed"]
            resources_used = building["resources_used_to_build_so_far"]
            for resource, count in resources_needed.items():
                # If the agent has the resource and the resource is still needed to build the building, we would use the resource
                if state["agents"][agent]["inventory"][resource] > 0 and resources_used[resource] < count:
                    building["resources_used_to_build_so_far"][resource] += 1
                    state["agents"][agent]["inventory"][resource] -= 1
                    worked_on_building = (building_id, building["type"])
                    break
            break
    
    # If a building was built, the action was successful
    if worked_on_building:
        # If the building has been completely built, we would update the layout of the world
        building = state["buildings"][worked_on_building[0]]
        building_cost = building["resources_needed"]
        if all([building_cost[resource] == building["resources_used_to_build_so_far"][resource]
                for resource in building_cost.keys()]):
            state["layout"][loc[0]][loc[1]] = type_to_layout[building["type"]]
            state["buildings"][worked_on_building[0]]["built"] = True
        
        # We would return the updated state and a flag indicating the action was successful
        return state, True

    # This point would only be reached if the building was not found at the location and the action was not successful
    return state, False

# We would define the state transition function
def state_transition(state, actions_dict, config):
    successful_actions = {agent: False for agent in actions_dict.keys()}

    # We would have a random order of agents for them to take actions
    agent_order = list(actions_dict.keys())
    random.shuffle(agent_order)

    for agent in agent_order:
        # We would get the action for the agent
        action = actions_dict[agent]

        # If we want the agent to move up
        if action == 0:
            state, successful_actions[agent] = move_agent(state, agent, "UP")
        
        # If we want the agent to move down
        elif action == 1:
            state, successful_actions[agent] = move_agent(state, agent, "DOWN")
        
        # If we want the agent to move left
        elif action == 2:
            state, successful_actions[agent] = move_agent(state, agent, "LEFT")
        
        # If we want the agent to move right
        elif action == 3:
            state, successful_actions[agent] = move_agent(state, agent, "RIGHT")
        
        # If we want the agent to collect the resouce
        elif action == 4:
            state, successful_actions[agent] = collect_resource(state, agent)
        
        # If we want the agent to build
        elif action == 5:
            state, successful_actions[agent] = build(state, agent)
        
        # If the action is invalid
        else:
            raise ValueError(f"Invalid action: {action}")
    
    # We would update the layout of the world
    state = set_layout(state, config)
    
    return state, successful_actions


def get_observations_from_state(state):
    """
    This function would return the observations for each agent based on the current state
    :param state: The current state of the world:
    :return: observations: A dictionary containing the observations for each agent
    """
    # The observations for the different agents would be as follows:
    # Agent parameters
        # The location (x, y) of the agent
        # The orientation of the agent
        # The inventory of the agent
        # The location of the other agents
    # Other Agent Parameters are appended after the agent parameters
    # Reosurce Tile Parameters
        # The location (x, y) of the resource tile
        # The type of the resource tile
        # The count of the resource tile
    # Building Parameters
        # The location (x, y) of the building
        # The type of the building
        # The resources used to build the building so far (wood, stone)
        # A flag indicating if the building has been built
    
    observations = {agent: [] for agent in state["agents"].keys()}
    for agent, agent_info in state["agents"].items():
        # We would append the agent location to the observation
        observations[agent].append(agent_info["location"][1]/100)
        observations[agent].append(agent_info["location"][0]/100)
        # We would append the agent orientation to the observation
        observations[agent].append(ORIENTATION[agent_info["orientation"]]/100)
        # We would append the agent inventory to the observation with wood first, the stone
        for resource in ["wood", "stone"]:
            observations[agent].append(agent_info["inventory"][resource]/100)

        # We would append the observations for the other agents first
        for other_agent, other_agent_info in state["agents"].items():
            if other_agent != agent:
                # We would append the other agent location to the observation
                observations[agent].append(other_agent_info["location"][1]/100)
                observations[agent].append(other_agent_info["location"][0]/100)
                # We would append the other agent orientation to the observation
                observations[agent].append(ORIENTATION[other_agent_info["orientation"]]/100)
                # We would append the other agent inventory to the observation with wood first, the stone
                for resource in ["wood", "stone"]:
                    observations[agent].append(other_agent_info["inventory"][resource]/100)
        
        # We would append the observations for the resource tiles
        tile_order = sorted(state["resource_tiles"].keys())
        for tile in tile_order:
            tile_info = state["resource_tiles"][tile]
            # We would append the tile location to the observation
            observations[agent].append(tile_info["location"][1]/100)
            observations[agent].append(tile_info["location"][0]/100)

            # We would append the tile type to the observation with 0.01 for wood, 0.02 for stone
            if tile_info["type"] == "wood":
                observations[agent].append(0.01)
            if tile_info["type"] == "stone":
                observations[agent].append(0.02)
            # We would append the tile count to the observation
            observations[agent].append(tile_info["count"]/100)

        if len(state["resource_tiles"]) < RESOURCE_TILES_COUNT:
            # We would append arrays of [0.0, 0.0, 0.0, 0.0] to the observation for the missing resource tiles
            for _ in range(RESOURCE_TILES_COUNT - len(state["resource_tiles"])):
                observations[agent].extend([0.0, 0.0, 0.0, 0.0])
        
        # We would append the observations for the buildings
        building_order = sorted(state["buildings"].keys())
        for building in building_order:
            building_info = state["buildings"][building]
            # We would append the building location to the observation
            observations[agent].append(building_info["location"][1]/100)
            observations[agent].append(building_info["location"][0]/100)

            # We would append the building type to the observation with 0.01 for house, 0.02 for mansion, 0.03 for castle
            if building_info["type"] == "house":
                observations[agent].append(0.01)
            if building_info["type"] == "mansion":
                observations[agent].append(0.02)
            if building_info["type"] == "castle":
                observations[agent].append(0.03)

            # We would append the resources used to build the building so far to the observation
            for resource in ["wood", "stone"]:
                observations[agent].append(building_info["resources_used_to_build_so_far"][resource]/100)

            # We would append 0.01 if the building has been built, 0.0 otherwise
            observations[agent].append(int(building_info["built"])/100)

        # We would append the arrays of [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] to the observation for the missing buildings
        if len(state["buildings"]) < BUILDINGS_COUNT:
            for _ in range(BUILDINGS_COUNT - len(state["buildings"])):
                observations[agent].extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return observations

# We would now define the reward function
def get_rewards(state, succesful_actions, step_cost, terminal_reward):
    rewards = {agent: step_cost for agent in succesful_actions.keys()}
    # We don't penalize the agents for taking an invalid action in the current iteration of the code
    
    # for agent, success in succesful_actions.items():
    #     if success:
    #         rewards[agent] = 0

    # If all the buildings have been built, we would give a terminal reward
    if all([building["built"] for building in state["buildings"].values()]):
        for agent in rewards:
            rewards[agent] += terminal_reward

    return rewards


# We would now define the MultiAgent Environment class
class MultiAgentCollectAndBuild(MultiAgentEnv):
    def __init__(self, t_config: EnvContext):
        super().__init__()
        self.random_seed = None
        self.config = t_config
        self.state = instantiate_world(self.config)
        self.step_results = {}
        self.episode_length = 0
        self.episode_reward = 0
        self.user_data_fields = []
        self.target_diagnostics = ["Total_Episode_Return", "Episode_Length", "Success", "Resources_Collected",
                                    "Total_Resources_Required", "Buildings_Built", "Fractional_Buildings_Built",
                                    "Total_Buildings"]
        self.observation_size = (5*len(self.state["agents"]) +
                                 4*RESOURCE_TILES_COUNT + 6*BUILDINGS_COUNT)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.observation_size,), dtype=np.float32)
        self.observation_space_dict = {agent: self.observation_space for agent in self.state["agents"].keys()}
        self.nA = len(ACTIONS)
        self.action_space = gym.spaces.Discrete(self.nA)
        self.action_space_dict = {agent: self.action_space for agent in self.state["agents"].keys()}
        self._agent_ids = list(self.state["agents"].keys())

    def reset(self):
        self.state = instantiate_world(self.config)
        self.episode_length = 0
        self.episode_reward = 0
        self.step_results = {}
        return get_observations_from_state(self.state)
    
    def step(self, actions_dict):
        # We would perform the state transition on the current state based on the actions taken by the agents
        self.state, successful_actions = state_transition(self.state, actions_dict, self.config)

        # We would get the observattions based on the current state
        observations = get_observations_from_state(self.state)

        # We would get the rewards based on the current state
        rewards = get_rewards(self.state, successful_actions, self.config["step_cost"], self.config["terminal_reward"])

        # If all the buildings have been built or we are over the horizon length, we would set the done flag to True
        dones = {"__all__": all([building["built"] for building in self.state["buildings"].values()]) \
                            or self.episode_length >= self.config["horizon"]}

        # We would set the dones for the individual agents
        for agent in self.state["agents"].keys():
            dones[agent] = dones["__all__"]
        
        # We would get the info for the step
        infos = {agent: {"episode_length": self.episode_length, "inventory": self.state["agents"][agent]["inventory"]} \
                 for agent in self.state["agents"].keys()}

        # We would add the percentage of each building completion
        building_comp = {building+1: sum([self.state["buildings"][building]["resources_used_to_build_so_far"][resource] \
                        for resource in self.state["buildings"][building]["resources_used_to_build_so_far"].keys()]) \
                        / sum([self.state["buildings"][building]["resources_needed"][resource] \
                        for resource in self.state["buildings"][building]["resources_needed"].keys()]) \
                        for building in self.state["buildings"].keys()}

        # The building comp info is added to the info for the agents
        for agent in self.state["agents"].keys():
            infos[agent]["building_completion"] = building_comp

        # We would update the episode length and reward
        self.episode_length += 1
        self.episode_reward += sum(rewards.values())

        # We would store the current step results
        self.step_results = {
            "observations": observations,
            "rewards": rewards,
            "dones": dones,
            "infos": infos
        }

        # We would return the observations, rewards, dones and infos
        return observations, rewards, dones, infos

    def render(self, mode="human"):
        if mode:
            for row in self.state["layout"]:
                print(row)
    
    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent: self.action_space.sample() for agent in agent_ids}
    

    def set_seed(self, seed, verbose=0):
        self.random_seed = seed
        random.seed(seed)
        if verbose:
            print(f"Environment seed set to {seed}")
    
    def get_diagnostics(self):
        # We would get the total episode return , episode length and success(All buildings built)
        diagnostics = {"Total_Episode_Return": self.episode_reward, "Episode_Length": self.episode_length,
                       "Success": all([building["built"] for building in self.state["buildings"].values()])}


        # We would get the resources collected by the agents so far
        # We need to subtract from the initial and current resource counts to get the resources collected
        initial_resources = sum([tile["count"] for tile in self.config["resource_tiles"]])
        current_resources = sum([tile["count"] for tile in self.state["resource_tiles"].values()])
        resources_required_for_buildings = sum([building["cost"]["wood"] + building["cost"]["stone"]
                                                for building in self.config["to_build"]])

        diagnostics["Resources_Collected"] = initial_resources - current_resources
        diagnostics["Total_Resources_Required"] = resources_required_for_buildings

        # We would get the number of buildings built
        diagnostics["Buildings_Built"] = sum([1 for building in self.state["buildings"].values()
                                              if building["built"]])

        # We would get the Fractional count  of buildings built from the current info
        diagnostics["Fractional_Buildings_Built"] = sum(self.step_results["infos"]["AGENT1"]\
                                                            ["building_completion"].values())

        diagnostics["Total_Buildings"] = len(self.state["buildings"])

        return diagnostics

register_env("MultiAgentCollectAndBuild", lambda config: MultiAgentCollectAndBuild(world_config))



if __name__ == "__main__":
    env = MultiAgentCollectAndBuild(world_config)
    print(env.reset())
    d = {"__all__": False}
    while not d["__all__"]:
        actions = env.action_space_sample()
        print(actions)
        o, r, d, i = env.step(actions)
        print(o, r, d, i, sep="\n")
        env.render()
        print(env.get_diagnostics())
        print("\n\n")