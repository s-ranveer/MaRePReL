import sys

import numpy as np
from utils.ma_plan_scheduler import aggregate_tasks, calculate_task_duration, create_distributed_plans
from multiagent_envs.dungeon.dungeon_methods import dungeon_task_aggregator
from gym.spaces import Box
from utils.planner import Planner
from utils import pyhop as hop

def achieve_goal(state, goals):
    """
    The method to check whether the goals have been achieved
    :param state: The current state of the environment
    :param goals: The goals to be achieved
    """
    goals_achieved = True
    unfulfilled_goals = []
    for key in goals.keys_required:
        # If the key is not in the door, we would add the key to the list of goals we
        # need to achieve
        if abs(state.door["key"][key] - 1) > 0.0001:
            goals_achieved = False
            unfulfilled_goals.append(key)

    if goals_achieved:
        return []

    return [("get_key", unfulfilled_goals[0]), ("achieve_goal", goals)]


def attackEnemy(state, enemy_id):
    """
    The operator to attack an enemy
    :param state: The current state of the environment
    :param enemy_id: The id of the enemy to be attacked
    """

    num_agents_alive = len([agent_id for agent_id, agent_details in state.agents.items()
                            if agent_details["health"] > 0])

    # We can only attack an alive enemy
    if state.enemies[enemy_id]["health"] > 0 or (len(state.enemies[enemy_id]["agitated"]) != num_agents_alive
                                                 and len(state.enemies[enemy_id]["agitated"]) != 0):
        # There needs to be a player available to attack the enemy
        for player_id, player_details in state.agents.items():
            # We would get the number of agents alive

            # We would check if the player is alive and we still have some enemies alive
            if player_details["health"] > 0:
                # If the player has not already attacked the enemy in this turn, we would attack the enemy
                if player_id not in state.enemies[enemy_id]["agitated"]:
                    # We would reduce the enemy's health by the player's attack
                    state.enemies[enemy_id]["health"] = max(0, state.enemies[enemy_id]["health"]
                                                               - player_details["attack"])
                    # The player would be added to the agitated list
                    state.enemies[enemy_id]["agitated"].append(player_id)

                    # num_agents_alive = len([agent_id for agent_id, agent_details in state.agents.items()
                    #                         if agent_details["health"] > 0])
                    if len(state.enemies[enemy_id]["agitated"]) == num_agents_alive:
                        state.enemies[enemy_id]["agitated"] = []
                        state.enemies[enemy_id]["health"] = 0

                    # The planner at the highest level is not really concerned with where the agent is
                    # We would just return the updated state
                    return state

    return False


def getKeyInDoor(state, key_id):
    """
    The operator to get the key in the door
    :param state: The current state of the environment
    :param key_id: The id of the key to be obtained
    """

    # The key either needs to be from an enemy which is no longer alive or already with a player,
    # and not in a door

    # If the key is not already in the door
    if np.abs(state.door["key"][key_id] - 1) > 0.0001:
        # The key is not with any agent
        if abs(state.enemies[key_id]["health"] - 0) < 0.0001:
            if abs(state.enemies[key_id]["key"] - 0) < 0.0001:
                # We would iterate over the agents and assign the key to the first agent which is alive if the key
                # is not already assigned to an agent
                for agent_id, agent_details in state.agents.items():
                    if agent_details["health"] > 0:
                        # The key number assigned is the agent number of the id divided by 100
                        state.enemies[key_id]["key"] = (int(agent_id.split("AGENT")[1]))/100

            # Once, the key has been assigned to the agent, the key is put in the door,
            # and the updated state is returned
            if np.abs(state.enemies[key_id]["key"] - 0) > 0.0001:
                state.door["key"][key_id] = 1
                state.enemies[key_id]["key"] = 0

                # We would iterate over all the keys in the door and check if they are all in the door
                all_keys_in_door = True
                for key, value in state.door["key"].items():
                    if np.abs(value - 1)> 0.0001:
                        all_keys_in_door = False
                        break

                # If all the keys are in the door, we would unlock the door
                if all_keys_in_door:
                    state.door["locked"] = 1

                return state

    return False



def add_unlock_with_key_2(enemy_key_id):
    """
    The method to add the unlock with key method
    :param enemy_key_id: The id of the enemy key
    :return: The unlock with key method
    """
    # The dynamic method which would consider case when we have to both
    # defeat the enemy and get the key in the door
    def attack_and_unlock_with_key_dynamic_method(state, key = enemy_key_id):
        """
        The method to attack and unlock with key
        :param state: The current state of the environment
        :param key: The id of the enemy key
        """
        tasks = []
        # We would check if the enemy is still alive
        if state.enemies[key]["health"] > 0:
            # We would return the task of defeating the enemy for every agent which is alive
            sorted_by_attacks_agents = sorted(state.agents.items(), key=lambda x: x[1]["attack"], reverse=True)
            for agent_id, agent_details in sorted_by_attacks_agents:
                if agent_details["health"] > 0:
                    tasks.append(("attackEnemy", key))
            tasks.append(("getKeyInDoor", key))
            return tasks
        return False
    attack_and_unlock_with_key_dynamic_method.__name__ = f"attack_and_unlock_with_key_{enemy_key_id}"
    return attack_and_unlock_with_key_dynamic_method


def add_unlock_with_key_1(enemy_key_id):
    """
    The method to add the unlock with key method
    :param enemy_key_id: The id of the enemy key
    """
    def unlock_with_key_dynamic_method(state, key = enemy_key_id):
        """
        The method to unlock with key
        :param state: The current state of the environment
        :param key: The id of the enemy key
        """
        # We would return the get key in door task if the enemy with the key is no longer alive
        if abs(state.enemies[key]["health"] - 0) < 0.0001:
            return [("getKeyInDoor", key)]
        return False

    unlock_with_key_dynamic_method.__name__ = f"unlock_with_key_{enemy_key_id}"
    return unlock_with_key_dynamic_method


def define_dynamic_methods(num_enemies):
    """
    The method to define the dynamic methods for the dungeon environment
    :param num_enemies: The number of enemies in the environment
    """
    dynamic_methods = []
    for enemy in range(1, num_enemies+1):
        dynamic_methods.append(add_unlock_with_key_1(enemy))
        dynamic_methods.append(add_unlock_with_key_2(enemy))
    hop.declare_methods("get_key", *dynamic_methods)


def declare_methods_and_operators(num_enemies):
    """
    The method to declare the methods and operators for the dungeon environment
    :param num_enemies: The number of enemies in the environment
    """
    hop.declare_methods("achieve_goal", achieve_goal)

    # We would define the dynamic methods for the dungeon environment
    define_dynamic_methods(num_enemies)

    # We would declare the operators for the dungeon environment
    hop.declare_operators(attackEnemy, getKeyInDoor)


def get_environment_state(obs, num_agents):
    """
    The method to get the environment state
    :param obs: The observation of the environment
    :param num_agents: The number of agents in the environment
    :return: The current state and the goal state for the office environment
    """

    # We get the number of agents from the environment
    agents = list(obs.keys())
    # We get the number of enemies from the environment
    num_enemies = int((obs[agents[0]].size - 3 - 6*num_agents)/4)

    # We would define the state for Pyhop
    state = hop.State("state")
    state.agents = {}
    state.enemies = {}
    state.door = {"key": {}, "location": None, "unlock": None}
    # We would iterate over the observations for the different agents
    for agent_id, agent_obs in obs.items():
        state.agents[agent_id] = {}
        # We would get the agent locations
        state.agents[agent_id]["location"] = (agent_obs[0], agent_obs[1])
        # We would get the agent's orientation
        state.agents[agent_id]["orientation"] = agent_obs[2]
        # We would get the agent's health
        state.agents[agent_id]["health"] = agent_obs[3]
        state.agents[agent_id]["attack"] = agent_obs[4]
        # We would get the agent's defense
        state.agents[agent_id]["defense"] = agent_obs[5]

        # We would get the x, y, hp, and key value for the different enemies
        for i in range(num_enemies):
            state.enemies[i+1] = {}
            state.enemies[i+1]["location"] = (agent_obs[6*num_agents + 4*i], agent_obs[6*num_agents + 4*i + 1])
            state.enemies[i+1]["health"] = agent_obs[6*num_agents + 4*i + 2]
            state.enemies[i+1]["key"] = agent_obs[6*num_agents + 4*i + 3]
            state.enemies[i+1]["agitated"] = []
            state.door["key"][i+1] = agent_obs[6*num_agents + 4*i + 3]

        # We would get the x, y and unlock value for the door
        state.door["location"] = (agent_obs[6*num_agents + 4*num_enemies], agent_obs[6*num_agents + 4*num_enemies + 1])
        state.door["locked"] = agent_obs[6*num_agents + 4*num_enemies + 2]

    # We also need to set the goal
    goal = hop.State("goal")
    goal.keys_required = []
    for key, value in state.door["key"].items():
        # If the key is not in the door, we would add the key to the goal
        if np.abs(value - 1) > 0.0001:
            goal.keys_required.append(key)

    return state, goal


class MultiAgentDungeonPlanner(Planner):
    def __init__(self, obs):
        """
        The planner for the dungeon environment
        :param obs: the observation of the environment
        """
        self.obs = obs
        self.original_obs_size = obs[list(obs.keys())[0]].size
        self.plan = None
        self.goal = None

        assert type(self.obs) == dict

        # The observation space will tell us the number of agents
        self.agents = list(self.obs.keys())
        self.all_agents = self.agents
        num_agents = len(self.agents)

        # The planner would exit with an error if there are no agents
        if num_agents < 1:
            sys.exit("There are no agents to plan for")
        else:
            # We need to calculate the number of enemies
            obs_size = self.obs[self.agents[0]].size
            self.num_enemies = int((obs_size - 3 - 6*num_agents)/4)

            declare_methods_and_operators(self.num_enemies)
            self.operators = hop.operators.keys()

    def get_agents(self):
        """
        The method to get the agents or players for the environment
        :return: The list of environment agents
        """
        return self.agents

    def get_operators(self):
        """
        The method to get the operators for the environment
        :return: The list of operators for the environment
        """
        return self.operators


    def reset(self, obs):
        """
        The reset method for the dungeon planner
        :param obs: The current observation state
        :return: None
        """
        self.obs = obs
        self.plan = None
        self.goal = None

        assert type(self.obs) == dict

        # The observation space will tell us the number of agents
        self.agents = list(self.obs.keys())
        num_agents = len(self.agents)

        # The planner would exit with an error if there are no agents
        if num_agents < 1:
            sys.exit("There are no agents to plan for")
        else:
            # We need to calculate the number of enemies
            obs_size = self.obs[self.agents[0]].size
            self.num_enemies = int((obs_size - 3 - 6*num_agents)/4)

            # We need to declare the methods and operators again
            declare_methods_and_operators(self.num_enemies)
            self.operators = hop.operators.keys()

    def next_tasks(self, obs, **kwargs):
        """
        The method to get the next tasks for the agents
        :param obs: The observation of the environment
        :param kwargs: The additional arguments to be provided
        :return: The next tasks for the agents
        """
        next_task = dict()
        return_all_tasks = False

        # The planner is a dictionary with a plan for each agent
        self.plan = self.get_plan(obs)

        # If no plan is found, we return an empty dictionary
        if self.plan is None:
            return {}

        # Append the first task for the current plan to the next tasks
        if kwargs and "return_all_tasks" in kwargs:
            return_all_tasks = kwargs["return_all_tasks"]

        for agent, plan in self.plan.items():
            if plan:
                if return_all_tasks:
                    next_task[agent] = plan
                else:
                    next_task[agent] = plan[0]
            else:
                next_task[agent] = None

        return next_task


    def is_task_done(self, obs, tasks):
        """
        The method to check whether the tasks have been done
        :param obs: The observation of the environment
        :param tasks: The tasks for the agents
        """
        # We need to check whether a task was completed by the agent.
        num_agents = len(self.all_agents)
        tasks_done = dict()

        # We would iterate over the agents and their tasks
        for agent, agent_obs in obs.items():
            if agent in tasks.keys():
                task = tasks[agent]

                # If the agent has a task assigned to it, we would check whether the task has been completed
                if task:
                    task_type, enemy_key_id = task
                    # If the task is attackEnemy, we would check whether the enemy is dead
                    if task_type == "attackEnemy":
                        # We would check whether the enemy is dead
                        if abs(agent_obs[6*num_agents + 4*(enemy_key_id-1) + 2] - 0) < 0.0001:
                            tasks_done[agent] = task_type
                        else:
                            tasks_done[agent] = False

                    # If the task is getKeyInDoor, we would check whether the key is in the door
                    if task_type == "getKeyInDoor":
                        # We would check whether the key is in the door
                        if abs(agent_obs[6*num_agents + 4*(enemy_key_id-1) + 3] - 1) < 0.0001:
                            tasks_done[agent] = task_type
                        else:
                            tasks_done[agent] = False

                else:
                    # If the agent has no task assigned to it, we would return True
                    tasks_done[agent] = None

        return tasks_done

    def is_plan_valid(self, obs, tasks):
        """
        The method to check whether the plan is valid
        :param obs: The observation of the environment
        :param tasks: The tasks for the agents
        :return: Whether the plan is valid
        """

        #Must add disrupted agents list
        # We need to check whether a task was completed by the agent.
        disrupted_agents = []
        num_agents = len(self.all_agents)
        disrupted_agents = []
        # We would iterate over the agents and their tasks
        for agent, agent_obs in obs.items():
            if agent in tasks.keys():
                task = tasks[agent]
                
                # If the agent has a task assigned to it, we would check whether the task has been completed
                if task:
                    
                    task_type, enemy_key_id = task
                    current_agent_index = self.all_agents.index(agent)

                    # If the task is getKeyInDoor, we would check whether the key is in the door
                    if task_type == "getKeyInDoor":
                        # if the key is not in the door and not picked up and not with the agent, we would return False
                        if abs(agent_obs[6*num_agents + 4*(enemy_key_id-1) + 3] - 0) > 0.0001 and \
                                abs(agent_obs[6*num_agents + 4*(enemy_key_id-1) + 3] - 1) > 0.0001 and \
                                abs(agent_obs[6*num_agents + 4*(enemy_key_id-1) + 3] -
                                    (current_agent_index+1)/100) > 0.0001:
                            disrupted_agents.append(agent)

                    # If the agent has a task assigned to it, but it is dead
                    if (agent_obs[3] - 0) < 0.0001:
                        disrupted_agents.append(agent)

        return len(disrupted_agents) == 0, disrupted_agents

    def get_abstract_obs(self, obs, tasks):
        """
        The method to get the abstract observation for the agents
        :param obs: The observation of the environment
        :param tasks: The tasks for the agents
        :return: The abstract observation for the agents
        """
        # In case of attack, we would have the agent's location, orientation, hp, attack, defense, and the enemy's
        # location, hp, key
        abstract_obs = dict()

        # We would get the number of agents
        num_agents = len(self.agents)
        num_enemies = self.num_enemies

        # We would iterate over the agents and their tasks
        for agent in tasks.keys():
            task = tasks[agent]

            # If the agent has a task assigned to it, we would get the task specific abstract observation
            if task:
                # Get the enemy or key id
                task_type, enemy_key_id = task

                if task_type == "attackEnemy":
                    # We would get the agent's location, orientation, hp, defense, and the enemy's
                    # location and hp
                    abstract_obs[agent] = np.concatenate((obs[agent][:4],
                                                          np.expand_dims(obs[agent][5], axis=0),
                                                          obs[agent][6*num_agents + 4*(enemy_key_id-1):
                                                                     6*num_agents + 4*(enemy_key_id-1) + 3]))

                if task_type == "getKeyInDoor":
                    # We would get the agent's location, orientation,  and the enemy's
                    # location, key
                    abstract_obs[agent] = np.concatenate((obs[agent][:3],
                                                                obs[agent][6*num_agents + 4*(enemy_key_id-1):
                                                                       6*num_agents + 4*(enemy_key_id-1) + 2],
                                                                np.expand_dims(obs[agent][6*num_agents +
                                                                                          4*(enemy_key_id-1) + 3],
                                                                               axis=0),
                                                                obs[agent][6*num_agents + 4*num_enemies:
                                                                           6*num_agents + 4*num_enemies+2]))
            else:
            # In case the agent has no assigned tasks, we would have the agent statistics along with all
            # remaining values being 0
                abstract_obs[agent] = np.concatenate((obs[agent][:4], np.zeros(4)))

        return abstract_obs

    def get_observation_space_dict(self, obs_dict, mode = "abstract"):
        """
        The method to get the observation space dict for the agents
        :param obs_dict: The observation space dict for the agents
        :param mode: The mode for the observation space
        :return: The observation space dict for the agents (The abstracted version)
        """
        for agent in self.agents:
            if mode == "abstract":
                obs_dict[agent] = Box(low=0, high=1, shape=(8, ), dtype=np.float32)
            elif mode == "extended":
                obs_dict[agent] = Box(low=-1, high=1, shape=(self.original_obs_size+4,), dtype=np.float32)
        return obs_dict


    def get_action_space_dict(self, action_dict):
        """
        The method to get the action space dict for the agents
        :param action_dict: The action space dict for the agents
        :return: The action space dict for the agents
        """
        return action_dict

    def get_plan(self, obs_dict):
        """
        The method to get the plans for the different agents
        :param obs_dict: The observation of the environment
        :return: The plan for the different agents
        """

        # We need to extract the goal from the environment
        self.agents = list(obs_dict.keys())
        state, goal = get_environment_state(obs_dict, num_agents = len(self.all_agents))

        # If there are no more goals remaining we would return None
        if len(goal.keys_required) == 0:
            return None

        # We would compute the single agent plan using Pyhop
        plan = hop.pyhop(state, [("achieve_goal", goal)], verbose=0)

        # Once we have the single agent plan, we would divide the plan between
        # the different agents
        if not plan:
            print(state, goal)
            print(obs_dict)
            sys.exit("No plan found. Try Again")

        # The tasks are aggregated so that we have the getKeyInDoor tasks where the same agent is holding the
        # keys for the task doing it together
        tasks = aggregate_tasks(plan, dungeon_task_aggregator, state=state)

        # We would calculate the task duration for the different tasks
        task_duration_dict = calculate_task_duration(tasks)

        # We would create the distributed plans for the different agents
        alive_agents = [agent_id for agent_id, agent_details in state.agents.items()
                                if agent_details["health"] > 0]
        distributed_plans = create_distributed_plans(task_duration_dict, len(alive_agents))
        # The plans outputted are somehow all lists. Fixing the issue here instead of rewriting the code
        # for create_distributed_plans
        for plan_id, plan in distributed_plans.items():
            p = [(plan[i], plan[i+1]) for i in range(0, len(plan), 2)]
            distributed_plans[plan_id] = p

        # We would assign the plans to the different agents
        tasks_to_reassign = []
        for plan_id, plan in distributed_plans.items():
            # We would check if there any agents which have two attackEnemy tasks in the plan for the same enemy
            # If there are, we would remove the second attackEnemy task, and assign it to the agent, which doesn't
            # have any attackEnemy task assigned to it

            # We would check for duplicate tasks
            for task in plan:
                # We would check if the task is in the plan more than once
                if plan.count(task) > 1 and (plan_id, task) not in tasks_to_reassign:
                    # We would add the task to the list of tasks to be reassigned
                    tasks_to_reassign.append((plan_id, task))

        # We would remove the tasks to be reassigned from their current plans, and reassign them
        for plan_id, task in tasks_to_reassign:
            distributed_plans[plan_id].remove(task)

            # Reassigning the task
            for p_id, p in distributed_plans.items():
                if task not in p:
                    distributed_plans[p_id].insert(0, task)
                    break


        # We need to get the tuples for the different plans and their corresponding getKeyInDoor tasks
        plan_key_tuples = []
        for plan_id, plan in distributed_plans.items():
            for task in plan:
                if task[0] == "getKeyInDoor":
                    plan_key_tuples.append((plan_id, task[1]))


        # We would iterate over the agents, assigning them plans corresponding to the keys they are holding
        agent_assigned_plans = {agent: None for agent in alive_agents}
        plans_assigned = []
        for agent in self.agents:
            for plan_id, key in plan_key_tuples:
                # An agent is assigned a plan only if it is alive
                i = self.all_agents.index(agent)
                if np.abs((i+1)/100 - state.enemies[key]["key"]) < 0.0001 and state.agents[agent]["health"] > 0:
                    agent_assigned_plans[agent] = distributed_plans[plan_id]
                    plans_assigned.append(plan_id)
                    break

        # We would assign the remaining plans to the agents who don't have any plans assigned to them
        for plan_id, plan in distributed_plans.items():
            if plan_id not in plans_assigned:
                for agent in self.agents:
                    # An agent is assigned a plan only if it is alive
                    if state.agents[agent]["health"] > 0:
                        if agent_assigned_plans[agent] is None:
                            agent_assigned_plans[agent] = plan
                            plans_assigned.append(plan_id)
                            break

        # We need to make sure that the attackEnemy tasks are executed first
        for agent, plan in agent_assigned_plans.items():
            if plan is not None:
                # We would sort the plan so that attackEnemy tasks are executed first
                plan.sort(key=lambda x: 0 if x[0] == "attackEnemy" else 1)
                agent_assigned_plans[agent] = plan

        # We need to sort the attackEnemy tasks in increasing order of enemy id for the same agent
        # Extra sorting to try improve the planner performance. We want the agents to attack the same enemies
        # together for maximum performance (maybe)
        for agent, plan in agent_assigned_plans.items():
            if plan is not None:
                # Extract 'attackEnemy' tuples and sort them
                attack_enemy_tuples = sorted((t for t in plan if t[0] == 'attackEnemy'), key=lambda x: x[1])

                # Replace the sorted 'attackEnemy' tuples in the original list
                agent_assigned_plans[agent] = [t if t[0] != 'attackEnemy' else attack_enemy_tuples.pop(0) for t in plan]

        # There is one final sorting we need to do. If there is a getKeyInDoor task, but no corresponding attackEnemy
        # task, we would sort the plan so that the getKeyInDoor task is executed first
        attackEnemyTaskforEnemyExists = {}
        for agent, plan in agent_assigned_plans.items():
            if plan is not None:
                for task in plan:
                    if task[0] == "attackEnemy":
                        attackEnemyTaskforEnemyExists[task[1]] = True

        # Once, we have the attackEnemyTaskforEnemyExists, we would sort the plan so that the getKeyInDoor task is
        # executed first in case there is no attackEnemy task for the enemy or it is already dead
        for agent, plan in agent_assigned_plans.items():
            if plan is not None:
                # We would sort the plan so that attackEnemy tasks are executed first
                plan.sort(key=lambda x: 0 if x[0] == "getKeyInDoor" and x[1] not in attackEnemyTaskforEnemyExists.keys()
                                         else 1)
                agent_assigned_plans[agent] = plan

        # We need to further sort the plans so that the attackEnemy tasks are executed in order of thr enemies in hand
        for agent, plan in agent_assigned_plans.items():
            if plan is not None:
                # We would sort the plan so that attackEnemy tasks are executed in order of the enemy
                attack_enemy_tuples = sorted((t for t in plan if t[0] == 'attackEnemy'), key=lambda x: x[1])
                plan = [t if t[0] != 'attackEnemy' else attack_enemy_tuples.pop(0) for t in plan]
                agent_assigned_plans[agent] = plan

        self.plan = agent_assigned_plans
        return agent_assigned_plans

    def get_current_plan(self):
        """
        The method to get the current plan
        :return: The current plan
        """
        return self.plan






