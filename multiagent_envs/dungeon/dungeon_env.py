import copy
import random
import sys
import gym
import os
import numpy as np
from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune import register_env

# These are the actions we define for the environment
ACTIONS = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
    "ATTACK": 4,
    "DEFEND": 5,
    "PICKUP": 6,
    "UNLOCK": 7
}

# Movements based on the direction
MOVEMENTS = {
    "LEFT": [0, -1],
    "RIGHT": [0, 1],
    "UP": [-1, 0],
    "DOWN": [1, 0]
}

# Orientations
ORIENTATION = {
    "LEFT": 0,
    "RIGHT": 1,
    "UP": 2,
    "DOWN": 3,
}
# This is how we would define the observation space for the environment
# For an Agent i, the observation space would be the
# x, y and orientation for all the agents.
# x, y and orientation for the dragon
# HP for all the agents (Can't go below 0
# HP for the dragon (Can't go below 1)
#


# This is the base config
# For the game layout, we have the following key for the layout
# X: Walls
# P: Pillars
# K: Key
# D: Door
# A: Agents
# T: Target (Dragon)

game_config = {
    "dungeon_layout": [
        ["X", "X", "X", "D", "X", "X", "X", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", "P", " ", " ", "P", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", "P", " ", " ", "P", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", " ", " ", " ", " ", " ", " ", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X"],
    ],
    "render": "human",
    "horizon": 500,
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 5,
        "unlock_door": 30,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 1,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 0,
            "attack": 1,
            "hp": 10,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 0,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}

def instantiate_game(config):
    """
    Method for initializing the game given the config
    :param config: The game config
    :return: The starting state for the game
    """
    state = dict()
    empty_locs = []
    door_loc = None
    layout = copy.deepcopy(config["dungeon_layout"])
    for row in range(len(layout)):
        r = layout[row]
        for col in range(len(r)):
            if layout[row][col] == " ":
                empty_locs.append([row, col])
            if layout[row][col] == "D":
                door_loc = [row, col]

    # Once, we have the empty rows, we would sample the locations for the different agents, and
    # we would sample the location for the dragon
    num_enemies = 0
    for enemy in config["enemies"]:
        num_enemies += config["enemies"][enemy]["count"]

    locs = random.sample(empty_locs, config["max_players"]+ num_enemies)
    agent_locs = locs[0:config["max_players"]]
    enemy_locs = {}

    # We would get the locations for the dragons, skeletons and wraiths
    if config["enemies"]["dragon"]["count"] > 0:
        dragon_locs = locs[config["max_players"]:config["max_players"]+config["enemies"]["dragon"]["count"]]
        enemy_locs["dragon"] = dragon_locs
        # We would add the dragons to the layout
        for dragon_loc in dragon_locs:
            layout[dragon_loc[0]][dragon_loc[1]] = "TD"

    # We would get the locations for the skeletons
    if config["enemies"]["skeleton"]["count"] > 0:
        skeleton_locs = locs[config["max_players"]+config["enemies"]["dragon"]["count"]:
                             config["max_players"]+config["enemies"]["dragon"]["count"]+
                             config["enemies"]["skeleton"]["count"]]
        enemy_locs["skeleton"] = skeleton_locs

        # We would add the skeletons to the layout
        for skeleton_loc in skeleton_locs:
            layout[skeleton_loc[0]][skeleton_loc[1]] = "TS"

    # We would get the location for the wraiths
    if config["enemies"]["wraith"]["count"] > 0:
        wraith_locs = locs[config["max_players"]+config["enemies"]["dragon"]["count"]+
                           config["enemies"]["skeleton"]["count"]:]
        enemy_locs["wraith"] = wraith_locs

        # We would add the wraiths to the layout
        for wraith_loc in wraith_locs:
            layout[wraith_loc[0]][wraith_loc[1]] = "TW"

    # Since, we have the location for the dungeon and the dragon, we can generate an environment instance
    for loc in agent_locs:
        layout[loc[0]][loc[1]] = "A"


    state["layout"] = layout
    state["grid_size"] = [len(layout), len(layout[0])]
    state["agents"] = dict()
    state["enemies"] = dict()
    state["dragon"] = dict()
    state["door_loc"] = door_loc
    state["door_locked"] = True
    state["keys_in_door"] = {}
    state["keys_available"] = {}

    # There are multiple keys in the game, each of which is available after killing an enemy
    for i in range(config["max_players"]):
        state["agents"][i] = dict()
        state["agents"][i]["loc"] = agent_locs[i]
        state["agents"][i]["hp"] = config["max_player_health"]
        state["agents"][i]["attack"] = config["player_attack"]
        state["agents"][i]["defend"] = False
        state["agents"][i]["key"] = []
        state["agents"][i]["orientation"] = 0

    for enemy, enemy_dict in config["enemies"].items():
        for i in range(enemy_dict["count"]):
            state["enemies"][f"{enemy}_{i}"] = dict()
            state["enemies"][f"{enemy}_{i}"]["loc"] = enemy_locs[enemy][i]
            state["enemies"][f"{enemy}_{i}"]["hp"] = enemy_dict["hp"]
            state["enemies"][f"{enemy}_{i}"]["attack"] = enemy_dict["attack"]
            state["enemies"][f"{enemy}_{i}"]["agitated"] = []
            state["enemies"][f"{enemy}_{i}"]["attack_next_turn"] = True
            state["enemies"][f"{enemy}_{i}"]["key"] = enemy_dict["key"]
            state["keys_available"][f"{enemy}_{i}"] = False
            state["keys_in_door"][f"{enemy}_{i}"] = False


    # We have initialized the game state and can return it
    return state

def set_layout(state, dungeon_layout):
    """
    Method for setting the layout for the state
    :param state: The current state of the game
    :param dungeon_layout: The layout of the dungeon
    """
    layout = copy.deepcopy(dungeon_layout)
    # We would set the layout for the state

    # We place the agents in the correct position
    for agent, agent_state in state["agents"].items():
        agent_loc = agent_state["loc"]
        # We would set the agent location in the layout only if it is alive
        if agent_state["hp"] > 0:
            layout[agent_loc[0]][agent_loc[1]] = "A"

    # We would set the enemy locations in the layout only if they are alive
    for enemy, enemy_state in state["enemies"].items():
        enemy_loc = enemy_state["loc"]
        if enemy_state["hp"] > 0:
            layout[enemy_loc[0]][enemy_loc[1]] = "T" + enemy.split("_")[0][0].upper()

    # We would set the key locations only if it is available and thd
    # We would set the key location in the layout only if it is available and the dragon is dead
    for key in state["keys_available"]:
        key_loc = state["enemies"][key]["loc"]
        if state["enemies"][key]["hp"] <= 0 and state["keys_available"][key]:
            layout[key_loc[0]][key_loc[1]] = "K"

    # We would set the door location in the layout only if it is locked
    if state["door_locked"]:
        door_loc = state["door_loc"]
        layout[door_loc[0]][door_loc[1]] = "D"

    state["layout"] = layout
    return state

def move(state, agent, direction):
    """
    The method to move the agent in a direction
    :param state: The current state of the game
    :param agent: The current agent
    :param direction: The direction to move the agent
    :return: The updated state and an indicator indicating whether the transition was successful and valid
    """
    # The success flag indicating whether the agent was able to take the action
    success = 0

    agent_loc = state["agents"][agent]["loc"]
    move_loc = [agent_loc[0] + MOVEMENTS[direction][0], agent_loc[1] + MOVEMENTS[direction][1]]

    # We have the new location. We would modify the state if it is possible to take the action
    # To be able to move to the location, we first need to check the locations are in the grid
    if not (move_loc[0] < 0 or move_loc[0] >= state["grid_size"][0]
            or move_loc[1] < 0 or move_loc[1] >= state["grid_size"][1]):
        # Since the values are in range, we need to check if the block is currently empty
        # We will swap the agent and the empty block

        if state["layout"][move_loc[0]][move_loc[1]] == " " or state["layout"][move_loc[0]][move_loc[1]] == "A":
            state["agents"][agent]["loc"] = move_loc
            success = 1

    # We change the orientation of the agent to the direction it is moving
    state["agents"][agent]["orientation"] = ORIENTATION[direction]

    # We have updated the state and can return it
    return state, success

def attack_agent(state, agent, friendly_fire_allowed):
    """
    Method for allowing the agent to attack
    :param state: The current game state
    :param agent: The agent who initiates the attack
    :param friendly_fire_allowed: Whether the agent can harm the other agents
    :return: The updated state and an integer code indicating the result of the action
    """
    agent_loc = state["agents"][agent]["loc"]
    agent_orient = state["agents"][agent]["orientation"]
    success = 0

    # We need to get the location the agent is facing based on the orientation
    reverse_orient_dict = {v: k for k, v in ORIENTATION.items()}
    attack_loc = [-1, -1]
    attack_loc[0] = agent_loc[0] + MOVEMENTS[reverse_orient_dict[agent_orient]][0]
    attack_loc[1] = agent_loc[1] + MOVEMENTS[reverse_orient_dict[agent_orient]][1]

    # We have the attack location, and now we need to check if it is valid
    if not (attack_loc[0] < 0 or attack_loc[0] >= state["grid_size"][0]
            or attack_loc[1] < 0 or attack_loc[1] >= state["grid_size"][1]):

        # If the attack location matches one of the enemies, we would attack the enemy
        for enemy_id, enemy_info in state["enemies"].items():
            if enemy_info["loc"] == attack_loc:
                success = 1
                state["enemies"][enemy_id]["hp"] = max(
                        state["enemies"][enemy_id]["hp"] - state["agents"][agent]["attack"], 0)

                if state["enemies"][enemy_id]["hp"] == 0:
                    # The enemy is now dead, and we can do nothing
                    # state["enemies"][enemy_id]["key"] = False
                    state["keys_available"][enemy_id] = True
                    success = 2
                    return state, success

                # We would add the agent to the agitated list of the enemy
                if agent not in state["enemies"][enemy_id]["agitated"]:
                    state["enemies"][enemy_id]["agitated"].append(agent)
                    state["enemies"][enemy_id]["time_to_attack"] = 1


        # We need to check for friendly fire
        if friendly_fire_allowed:
            for agent_id, agent_info in state["agents"].items():
                if agent_info["loc"] == attack_loc:
                    success = -1

                    # If the other agent can defend, it will defend the attack
                    if agent_info["defend"]:
                        state["agents"][agent_id]["defend"] = False
                    else:
                        state["agents"][agent_id]["hp"] = max(
                            state["agents"][agent_id]["hp"] - state["agents"][agent]["attack"], 0)
                        if state["agents"][agent_id]["hp"] == 0:
                            # The agent is now dead, and we can do nothing, but we need to deal with its death
                            # in case its holding any keys
                            state = agent_with_key_death(state, agent_id)
                            success = -2

                    return state, success

    # For any other kind of action, nothing changes, so we return the same state and 0 for success
    return state, success

def attack_by_enemy(state):
    """
    Method for allowing the enemies to attack the agent
    :param state: The current game state
    :return: The resulting state
    """
    # The dragon would attack only if it is alive and is agitated
    for enemy, enemy_state in state["enemies"].items():
        if enemy_state["hp"] > 0 and len(enemy_state["agitated"]) > 0:
            if enemy_state["time_to_attack"] == 0:
                for agent in enemy_state["agitated"]:
                    agent_loc = state["agents"][agent]["loc"]
                    enemy_loc = enemy_state["loc"]
                    # If the enemy is next to the agent, it would attack the agent
                    if abs(agent_loc[0] - enemy_loc[0]) + abs(agent_loc[1] - enemy_loc[1]) == 1:
                        # We would reduce the agent's hp by the enemy's attack power
                        state["agents"][agent]["hp"] = max(state["agents"][agent]["hp"] - enemy_state["attack"], 0)

                        if state["agents"][agent]["hp"] == 0:
                            # If the agent's hp is 0, we need to handle the agent's death. We need to free up
                            # the keys held by the agent
                            state = agent_with_key_death(state, agent)

                # The enemy is no longer agitated against the agent
                enemy_state["agitated"] = []
                enemy_state["time_to_attack"] = 1
            else:
                enemy_state["time_to_attack"] -= 1

    return state

def agent_with_key_death(state, agent):
    """
    Method for handling the death of the agent. If the agent has one or more key, we need to drop those keys
    in the agent's location or somewhere else.
    """
    # We would check if the agent has any key
    if state["agents"][agent]["key"]:
        # The key is now available to be picked up.
        # We could think as of having new keys available after defeating the enemy at the location of agent's death
        # and those around it
        for key in state["agents"][agent]["key"]:
            state["keys_available"][key] = True
            state["enemies"][key]["loc"] = state["agents"][agent]["loc"]

        # We would remove the keys from the agent
        state["agents"][agent]["key"] = []

    return state
def defend(state, agent):
    """
    Method for allowing the agent to defend
    :param state: The current game state
    :param agent: The agent who initiates the attack
    :return: The updated state and an integer code indicating the result of the action
    """
    # If we are not in a defence position, we toggle it. Otherwise, no action is taken
    if not state["agents"][agent]["defend"]:
        state["agents"][agent]["defend"] = True
        success = 1
    else:
        success = 0

    return state, success

def pickup(state, agent):
    """
    Method for allowing the agent to pick up the key
    :param state: The current game state
    :param agent: The agent trying to pick up the key
    :return: The updated state and an integer code indicating the result of the action
    """
    # We would check if there is a key in the block adjacent to the agent and the agent is facing the block
    agent_loc = state["agents"][agent]["loc"]
    agent_orient = state["agents"][agent]["orientation"]
    success = 0
    reverse_orient_dict = {v: k for k, v in ORIENTATION.items()}
    pickup_loc = [-1, -1]
    pickup_loc[0] = agent_loc[0] + MOVEMENTS[reverse_orient_dict[agent_orient]][0]
    pickup_loc[1] = agent_loc[1] + MOVEMENTS[reverse_orient_dict[agent_orient]][1]
    if not (pickup_loc[0] < 0 or pickup_loc[0] >= state["grid_size"][0]
            or pickup_loc[1] < 0 or pickup_loc[1] >= state["grid_size"][1]):

        # We would iterate over the keys and look for one which matches the location
        for key, key_available in state["keys_available"].items():
            if key_available:
                if state["enemies"][key]["loc"] == pickup_loc:
                    state["agents"][agent]["key"].append(key)
                    state["keys_available"][key] = False
                    success = 1

    return state, success

def unlock(state, agent):
    """
    Method for allowing the agent to unlock the door
    :param state: The current game state
    :param agent: The agent who will try to unlock the door
    :return: The current state and the success code
    """
    # We would check if the agent has the key and is facing the door
    agent_loc = state["agents"][agent]["loc"]
    agent_orient = state["agents"][agent]["orientation"]
    success = 0
    reverse_orient_dict = {v: k for k, v in ORIENTATION.items()}
    unlock_loc = [-1, -1]
    unlock_loc[0] = agent_loc[0] + MOVEMENTS[reverse_orient_dict[agent_orient]][0]
    unlock_loc[1] = agent_loc[1] + MOVEMENTS[reverse_orient_dict[agent_orient]][1]
    if not (unlock_loc[0] < 0 or unlock_loc[0] >= state["grid_size"][0]
            or unlock_loc[1] < 0 or unlock_loc[1] >= state["grid_size"][1]):
        # The location is valid, but now we need to check what are we exactly hitting
        # if state["door_loc"] == unlock_loc and state["agents"][agent]["key"]:
        #     state["agents"][agent]["key"] = False
        #     state["door_locked"] = False
        #     success = 1

        # The location is valid, we need to be sure that the location is that of a door
        if state["layout"][unlock_loc[0]][unlock_loc[1]] == "D":
            # We would iterate over the keys in the door
            for key, key_in_door in state["keys_in_door"].items():
                # if the key is already in door, we do not consider it
                if not key_in_door:
                    if key in state["agents"][agent]["key"]:
                        state["agents"][agent]["key"].remove(key)
                        state["keys_in_door"][key] = True
                        success = 1

            # We have iterated over all the keys in the door, and we need to check if the door is unlocked
            if all(state["keys_in_door"].values()):
                state["door_locked"] = False
                success = 2

    return state, success

def state_transitions(state, actions, config):
    # The agent actions are taken in random order. If we implement something like agility, the ordering
    # can be determined
    terminal = {a: False for a in state["agents"].keys()}
    terminal["__all__"] = False

    failure = False
    success = {a: 0 for a in state["agents"].keys()}
    agent_ordering = list(actions.keys())
    random.shuffle(agent_ordering)

    # The agents are randomly ordered so tie braking is done by the virtue of the agent going first
    for agent in agent_ordering:
        # We would get the action for the agent
        action = actions[agent]

        # If the agent is not alive, HP == 0, we do not perform any action for the agent
        if state["agents"][agent]["hp"] == 0:
            success[agent] = 0
            continue

        # If the action is move up, we would try to move the agent to the left y, x -> y, x-1
        if action == 0:
            state, success[agent] = move(state, agent, "UP")
            # If the action is move down, we would try to move the agent to the right y, x -> y, x+1
        elif action == 1:
            state, success[agent] = move(state, agent, "DOWN")
        # If the action is move left, we would try to move the agent to the up y, x -> y-1, x
        elif action == 2:
            state, success[agent] = move(state, agent, "LEFT")
        # If the action is move right, we would try to move the agent to the up y, x -> y-1, x
        elif action == 3:
            state, success[agent] = move(state, agent, "RIGHT")
        # If the action is attack, we would get the resulting state after the agent attempted to attack
        elif action == 4:
            state, success[agent] = attack_agent(state, agent, config["friendly_fire"])
        # If the action is defend, we would set the defend to True for the agent
        elif action == 5:
            state, success[agent] = defend(state, agent)
        # If the action is pickup, we try to pick up the key if it is facing the agent
        elif action == 6:
            state, success[agent] = pickup(state, agent)

        # If the action is unlock, we try to unlock the door if the agent has the key
        elif action == 7:
            state, success[agent] = unlock(state, agent)
            # If the door is unlocked, the game has reached the terminal state, and we can return the state
            if not state["door_locked"]:
                terminal = {a: True for a in state["agents"].keys()}
                terminal["__all__"] = True

        # The action taken is invalid, this part of the code should never be reached
        else:
            raise ValueError("Invalid action taken by the agent")

        # In case, we reached the terminal state, we can return the state as the game has ended
        if terminal["__all__"]:
            return state, success, terminal, failure

    # We are now in the enemy phase, and we would allow the enemies to attack the agents
    state = attack_by_enemy(state)
    state = set_layout(state, config["dungeon_layout"])

    # We would check if all the agents are dead. In such a case, the game is over
    all_agents_dead = True
    for agent in state["agents"].keys():
        if state["agents"][agent]["hp"] > 0:
            all_agents_dead = False
        else:
            terminal[agent] = True

    if all_agents_dead:
        terminal["__all__"] = True
        failure = True

    return state, success, terminal, failure

def get_observation(state, max_enemies=2):
    """
    Method for getting the observation for the different agents
    :param state: The current game state
    :return: The observation for the different agents
    """
    # x, y, orientation, hp, attack, defend for each agent with the stats for the agent being observed first
    # x, y, hp, key value for the different enemies
    # x, y of the door
    # 1 if the door is locked, 0 otherwise
    obs = dict()
    agents = list(state["agents"].keys())

    # We need to sort the agents by their id
    agents.sort()

    for agent in agents:
        # Observation for each agent is a vector of the following
        # x, y, orientation, hp, defend for the agent
        obs[agent] = [state["agents"][agent]["loc"][1]/100, state["agents"][agent]["loc"][0]/100,
                      state["agents"][agent]["orientation"]/100, state["agents"][agent]["hp"]/100]
        obs[agent].append(state["agents"][agent]["attack"]/100)
        obs[agent].append(1 if state["agents"][agent]["defend"] else 0)

        # x, y, orientation, hp, defend for the other agents
        for other_agent in agents:
            if other_agent != agent:
                obs[agent].append(state["agents"][other_agent]["loc"][1]/100)
                obs[agent].append(state["agents"][other_agent]["loc"][0]/100)
                obs[agent].append(state["agents"][other_agent]["orientation"]/100)
                obs[agent].append(state["agents"][other_agent]["hp"]/100)
                obs[agent].append(state["agents"][other_agent]["attack"] / 100)
                obs[agent].append(state["agents"][other_agent]["defend"])

        # x, y, hp, their key for the enemies
        for enemy in state["enemies"].keys():
            obs[agent].append(state["enemies"][enemy]["loc"][1]/100)
            obs[agent].append(state["enemies"][enemy]["loc"][0]/100)
            obs[agent].append(state["enemies"][enemy]["hp"]/100)


            # We would output 1 if the key is in the door, agent id if the agent has the key, 0 otherwise
            if state["enemies"][enemy]["key"]:
                key_in_door = False
                agent_has_key = False
                if state["keys_in_door"][enemy]:
                    obs[agent].append(1)
                    key_in_door = True

                # If the key is not in the door, we would output the id for the agent
                if not key_in_door:
                    for a in agents:
                        if enemy in state["agents"][a]["key"]:
                            obs[agent].append((a+1) / 100)
                            agent_has_key = True
                            break

                # If the agent does not have the key, we would output 0
                if not agent_has_key and not key_in_door:
                    obs[agent].append(0)
            else:
                obs[agent].append(0)
            
            if len(state["enemies"]) < max_enemies:
                total_current_enemies = len(state["enemies"])
                for i in range(max_enemies - total_current_enemies):
                    obs[agent].extend([0, 0, 0, 0])

        # x, y and lock status of the door
        obs[agent].append(state["door_loc"][1]/100)
        obs[agent].append(state["door_loc"][0]/100)
        obs[agent].append(1 if state["door_locked"] else 0)
        obs[agent] = np.array(obs[agent], dtype=np.float32)

    return obs

# We need to define the reward functions for the different agents. There is a -1 step cost for each agent
# regardless of the action taken. The reward for the agent is the sum of the following
def get_rewards(actions, success, failure, rewards_config):
    """
    Method for getting the rewards for the different agents
    :param actions: The actions taken by the agents
    :param success: The success codes for the different agents
    :param failure: Whether the game has reached the terminal state due to failure
    :param rewards_config: The reward dictionary
    :return: The rewards for the different agents
    """

    # The rewards config is enabled by default
    if "enabled" not in rewards_config.keys():
        rewards_config["enabled"] = True

    rewards = {agent: -0.1 for agent in actions.keys()}
    for agent, action in actions.items():
        # We default to the base rewards (-1) for the agents regardless of success or action taken
        if action == 4:
            # If the action was attack on the enemy, we get a reward if the enemy was attacked
            if success[agent] == 1 and rewards_config["enabled"]:
                rewards[agent] = rewards_config["attack_enemy"]

            # All the agents also get the reward when the enemy is killed
            if success[agent] == 2 and rewards_config["enabled"]:
                for other_agent in actions.keys():
                    rewards[other_agent] = rewards_config["kill_enemy"]

            # If the action was attack on the agent, but we didn't kill it, we get a negative reward
            if success[agent] == -1 and rewards_config["enabled"]:
                rewards[agent] = rewards_config["attack_agent"]

            # If the action was attack on the agent, and we killed it, we get a negative reward
            if success[agent] == -2 and rewards_config["enabled"]:
                rewards[agent] = rewards_config["kill_agent"]

        # If the action was pickup, we get a reward if we picked up the key
        if action == 6:
            if success[agent] == 1 and rewards_config["enabled"]:
                rewards[agent] = rewards_config["pickup"]

        # If the action was unlock, all the agents get a reward if the door is unlocked
        if action == 7:
            if success[agent] == 1 and rewards_config["enabled"]:
                for other_agent in actions.keys():
                    rewards[other_agent] = rewards_config["unlock_key"]

            # If the action was unlock, all the agents get a reward if the door is unlocked
            if success[agent] == 2:
                for other_agent in actions.keys():
                    rewards[other_agent] = rewards_config["unlock_door"]

        # If we reached a failure state (all agents are killed), all the agents get a negative reward
        if failure and rewards_config["enabled"]:
            for other_agent in actions.keys():
                rewards[other_agent] += rewards_config["failure"]

    return rewards


# We would define the environment class for the dungeon environment
class DungeonEnv(MultiAgentEnv):
    def __init__(self, t_config: EnvContext):
        # We would initialize the game
        super().__init__()
        self.config = t_config

        # Set the seed for the numpy environment
        if not t_config.get("seed"):
            self.random_seed = int(np.random.random())
        else:
            self.random_seed = t_config.get("seed")

        self.target_diagnostics = ["Total_Episode_Return", "Episode_Length", "Success", "Number_of_Agent_Deaths",
                                   "Number_of_Enemy_Kills", "Number_of_Keys_in_Door", "Total_Keys", "Total_Enemies",
                                   "Total_Agents"]
        self.horizon = t_config["horizon"]
        self.episode_length = 0
        self.user_data_fields = {}
        self.dead_agents = []
        self.state = instantiate_game(t_config)
        self.cell_size = 100
        self.h = len(self.state["layout"])
        self.w = len(self.state["layout"][0])
        self.episode_reward = 0 
        observation_size = 6*len(self.state["agents"]) + 4*len(self.state["enemies"]) + 3
        self.observation_size = observation_size
        self.agents = [f"AGENT{agent}" for agent in range(1, t_config["max_players"] + 1)]
        self.observation_space = gym.spaces.Box( low=0, high=1, shape=(observation_size, ), dtype=np.float32)
        self.observation_space_dict = gym.spaces.Dict({agent: self.observation_space for agent in self.agents})
        self.step_reward = None
        self.nA = len(ACTIONS)
        self.action_space =  gym.spaces.Discrete(self.nA)
        self.action_space_dict = gym.spaces.Dict({agent : self.action_space for agent in self.agents})
    # self.previous_step = dict()
        self._agent_ids = self.agents
        self.terminal = {agent: False for agent in self.agents}
        self.terminal["__all__"] = False
        self.failure = False
        if "render" in t_config.keys():
            if t_config["render"] == "pygame":
                import pygame
                pygame.init()
                self.screen = pygame.display.set_mode((self.h * self.cell_size, self.w * self.cell_size))
                self.__init__sprites()

    def __init__sprites(self):
        if "pygame" not in sys.modules:
            import pygame
        import glob
        self.render_order = ["wall", "tile", "pillar", "key", "door", "agent", "dragon", "skeleton", "wraith"]
        asset_path = '/'.join(os.path.realpath(__file__).split('/')[:-1] + ['sprites/*.png'])


        asset_paths = glob.glob(asset_path)
        self.sprite_letter = dict(zip(self.render_order, ["X", " ", "P", "K", "D", "A", "TD", "TS", "TW"], ))
        self.sprites = {self.sprite_letter[asset.split('/')[-1].split('.')[0]]: pygame.image.load(asset) for asset in
                        asset_paths}

        for key, sprite in self.sprites.items():
            self.sprites[key] = pygame.transform.scale(sprite, (self.cell_size, self.cell_size))
        self.sprites_loaded = True

    def set_seed(self, seed, verbose=False):
        """
        Method to set the seed for the environment
        """
        if verbose:
            print("Seed Set to : ", seed)
        self.random_seed = seed

    def get_diagnostics(self):
        """
        Method for getting the diagnostics data for the environment
        :return: The diagnostics data for the environment
        """
        diagnostics = {}

        # We would get the total return for the episode
        total_return = sum(list(self.step_reward.values()))
        diagnostics["Total_Episode_Return"] = self.episode_reward

        # We would get the length of the episodes
        diagnostics["Episode_Length"] = self.episode_length

        # We would get the overall success which happens when all the keys are in the door

        diagnostics["Success"] =  not self.state["door_locked"]

        # We would get the number of agents who have died
        num_deaths = 0
        for agent in self.agents:
            if self.state["agents"][int(agent.split("AGENT")[1])-1]["hp"] == 0:
                num_deaths += 1
        diagnostics["Number_of_Agent_Deaths"] = num_deaths

        # We would get the number of enemies who have died as the number of kills
        num_kills = 0
        for enemy in self.state["enemies"]:
            if self.state["enemies"][enemy]["hp"] == 0:
                num_kills += 1

        diagnostics["Number_of_Enemy_Kills"] = num_kills
        diagnostics["Total_Keys"] = len(self.state["enemies"])
        diagnostics["Total_Enemies"] = len(self.state["enemies"])
        diagnostics["Total_Agents"] = len(self.agents)

        # We would get the number of keys in the door
        num_keys_in_door = 0
        for key in self.state["keys_in_door"]:
            if self.state["keys_in_door"][key]:
                num_keys_in_door += 1

        diagnostics["Number_of_Keys_in_Door"] = num_keys_in_door

        return diagnostics

    def reset(self, **kwargs):
        # We would reset the environment
        np.random.seed(int(self.random_seed))
        self.state = instantiate_game(self.config)
        self.episode_reward = 0
    # self.previous_step = dict()
        self.dead_agents = []
        self.terminal = {agent: False for agent in self.agents}
        self.terminal["__all__"] = False
        self.failure = False
        self.episode_length = 0
        observations = get_observation(self.state)
        # Map the keys of the observation to their equivalent for the environment
        return {f"AGENT{k+1}": observations[k] for k in observations.keys()}

    def step(self, actions):
        # We would take the actions for the different agents
        # We need to refactor the keys of the actions to match the keys of the agents for the state_transitions
        # AGENT1 maps to 0, AGENT2 maps to 1 and so on in case the keys are not integers
    # try:
    #     self.previous_step = dict()
    #     self.previous_step["initial_state"] = self.state
    #     self.previous_step["dead_agents_before_step"] = self.dead_agents
    # self.previous_step["actions"] = actions
            if not isinstance(list(actions.keys())[0], int):
                actions = {int(agent.split("AGENT")[1])-1: actions[agent] for agent in actions.keys()}

            self.state, success, terminal, self.failure = state_transitions(self.state, actions, self.config)
    # self.previous_step["successor_state"] = self.state
            self.episode_length += 1
            rewards = get_rewards(actions, success, self.failure, self.config["rewards"])
            observations = get_observation(self.state)

            # We need to map the keys of the observation, reward, terminal, and info
            # to their equivalent for the environment
            observations = {f"AGENT{k+1}": observations[k] for k in observations.keys()
                            if f"AGENT{k+1}" not in self.dead_agents}
            rewards = {f"AGENT{k+1}": rewards[k] for k in rewards.keys()
                       if f"AGENT{k+1}" not in self.dead_agents}
            self.step_reward = rewards
            self.episode_reward = self.episode_reward + sum(rewards.values())
            done = {f"AGENT{k+1}": terminal[k] for k in terminal.keys() if type(k) == int and
                    f"AGENT{k+1}" not in self.dead_agents}
            done["__all__"] = terminal["__all__"]

            if self.episode_length >= self.horizon:
                for key in done.keys():
                    done[key] = True

            self.terminal = done
            # self.previous_step["dones"] = self.terminal
            # self.previous_step["failure"] = self.failure

            info = {agent: {"Episode Length": self.episode_length, "Alive": self.state["agents"][i]["hp"] > 0}
                    for i, agent in enumerate(self.agents) if agent not in self.dead_agents}
            self.dead_agents = [agent for agent in self.agents
                                if self.state["agents"][int(agent.split("AGENT")[1])-1]["hp"] == 0]
            # self.previous_step["dead_agents_after_step"] = self.dead_agents
            return observations, rewards, done, info

        # except:
        #     print("Error in step. Previous step was")
        #     print(self.previous_step)




    def display_game(self, wait_time=200):
        """
        Method for displaying the game using pygame
        :param wait_time: The time to wait before displaying the next frame
        :return: None
        """
        self.screen.fill((0, 0, 0))
        # We would load the sprites for the different objects
        for row in range(len(self.state["layout"])):
            for col in range(len(self.state["layout"][row])):
                self.screen.blit(self.sprites[self.state["layout"][row][col]], (row * 100, col * 100))

        # We would update the display
        if "pygame" not in sys.modules:
            import pygame

        pygame.display.update()
        pygame.time.wait(wait_time)

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        actions = {}
        agents = self.get_agent_ids()

        for a in agents:
            actions[a] = random.randrange(0, self.nA - 1)
        return actions

    def render(self, mode=None):
        if "render" in self.config.keys():
            if self.config["render"] == "human":
                for line in self.state["layout"]:
                    print(line)
            elif self.config["render"] == "pygame":
                self.__init__sprites()
                self.display_game()

    @property
    def get_agent_ids(self):
        return self.agents
        
    
register_env("dungeon_env", lambda config: DungeonEnv(config))
if __name__ == "__main__":
    env = DungeonEnv(game_config)
    o = env.reset()
    print(f"INIT STATE: {o}")
    env.render()
    for i in range(100):
        actions = input("Enter the actions of the three agents seperated by a space\n")
        actions = actions.split(" ")
        actions = {i: int(actions[i]) for i in range(len(actions))}
        o, r, d, info = env.step(actions)
        print(f"Actions: {actions}")
        print(f"Obs: {o}")
        print(f"Reward: {r}")
        print(f"Success: {d}")
        print(f"Info: {info}")
        print(env.get_diagnostics())
        env.render()
        print(r)
        if d["__all__"]:
            break