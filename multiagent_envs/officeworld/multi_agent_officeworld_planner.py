# I am not really that well versed in the code for this environment
import sys

import numpy as np
from multiagent_envs.officeworld.officeworld_methods import office_task_aggregator
from utils.ma_plan_scheduler import aggregate_tasks, calculate_task_duration, create_distributed_plans
from gym.spaces import Box
from utils.planner import Planner
from utils import pyhop as hop

VISITED_A = 0
VISITED_B = 1
VISITED_C = 2
VISITED_D = 3
HAS_MAIL = 4
HAS_COFFEE = 5
VISITED_OFFICE = 6
DELIVERED_MAIL = 7
DELIVERED_COFFEE = 8

goal_dict = {
    0: VISITED_A,
    1: VISITED_B,
    2: VISITED_C,
    3: VISITED_D,
    4: HAS_MAIL,
    5: HAS_COFFEE,
    6: VISITED_OFFICE,
    7: DELIVERED_MAIL,
    8: DELIVERED_COFFEE,
}



    
def achieve_goal(state, goal):
    """
    The method to check whether the goals have been achieved
    :param state: the current state
    :param goal: The goals we want to achieve
    """
    goals_achieved = True
    unfulfilled_goals = []
    for task in goal.tasks:
        if task not in state.tasks_completed:
            goals_achieved = False
            unfulfilled_goals.append(task)
    if goals_achieved:
        return []

    # We need to sort the goals so that the visited office is at the end of the list
    if VISITED_OFFICE in unfulfilled_goals:
        unfulfilled_goals.remove(VISITED_OFFICE)
        unfulfilled_goals.append(VISITED_OFFICE)

    return [('solve', unfulfilled_goals[0]), ("achieve_goal", goal)]


def visitOrPickup(state, obj):
    """
    The method for defining the dynamics of the pickup operator. The objects, which includes location checkmarks
    and mail/coffee, are collected when the location is visited
    :param state: The current state of the environment
    :param obj: The object to be picked up
    """
    # If the object/ checkpoint is the office location, then we would check if the agent has mail or coffee
    if obj == VISITED_OFFICE:

        # We would check if the agent has mail or coffee
        # If the agent has mail or coffee, then we would remove the mail or coffee from the agent's inventory
        # and add the corresponding task to the tasks completed
        for agent, inventory in state.agent_inventory.items():
            if HAS_MAIL in state.tasks_completed and inventory["Mail"]:
                state.agent_inventory[agent]["Mail"] = 0
                state.tasks_completed.remove(HAS_MAIL)
                state.tasks_completed.add(DELIVERED_MAIL)
                break
            if HAS_COFFEE in state.tasks_completed and inventory["Coffee"]:
                state.agent_inventory[agent]["Coffee"] = 0
                state.tasks_completed.remove(HAS_COFFEE)
                state.tasks_completed.add(DELIVERED_COFFEE)
                break

        state.tasks_completed.add(VISITED_OFFICE)
        return state

    if obj in [HAS_MAIL, HAS_COFFEE]:
        success = False
        obj_already_assigned = False
        for agent, inventory in state.agent_inventory.items():
            if obj == HAS_MAIL and inventory["Mail"]:
                obj_already_assigned = True
            if obj == HAS_COFFEE and inventory["Coffee"]:
                obj_already_assigned = True

        for agent, inventory in state.agent_inventory.items():
            if not obj_already_assigned:
                if obj == HAS_MAIL and inventory["Mail"] == 0:
                    state.agent_inventory[agent]["Mail"] = 1
                    state.tasks_completed.add(obj)
                    success = True
                    break
                if obj == HAS_COFFEE and inventory["Coffee"] == 0:
                    state.agent_inventory[agent]["Coffee"] = 1
                    state.tasks_completed.add(obj)
                    success = True
                    break

        if success:
            return state

    if obj in [VISITED_A, VISITED_B, VISITED_C, VISITED_D]:
            state.tasks_completed.add(obj)
            return state

    return False

def deliver(state, obj, delivered_obj):
    """
    The method for defining the dynamics of the  operator for delivery.
    :param state: The current state of the environment
    :param obj: The current task
    :param delivered_obj: The task completed
    """
    if obj in state.tasks_completed:
        for agent in state.agent_inventory:
            if obj == HAS_COFFEE and state.agent_inventory[agent]["Coffee"]:
                state.agent_inventory[agent]["Coffee"] = 0
                break
            if obj == HAS_MAIL and state.agent_inventory[agent]["Mail"]:
                state.agent_inventory[agent]["Mail"] = 0
                break
        state.tasks_completed.remove(obj)
        state.tasks_completed.add(delivered_obj)
        if delivered_obj in [DELIVERED_COFFEE, DELIVERED_MAIL]:
            state.tasks_completed.add(VISITED_OFFICE)
        return state
    return False

def add_deliver(has_predicate, deliver_predicate):
    def deliver_dynamic_method(state, obj):
        if obj == deliver_predicate:
            return [('visitOrPickup', has_predicate), ('deliver', has_predicate, obj)]
        return False

    deliver_dynamic_method.__name__ = f"deliver_{has_predicate}_{deliver_predicate}"
    return deliver_dynamic_method

def add_deliver_2(has_predicate, deliver_predicate):
    def deliver_dynamic_method(state, obj):
        if obj == deliver_predicate:
            for agent, inventory in state.agent_inventory.items():
                if ((inventory["Mail"] and has_predicate == HAS_MAIL)
                        or (inventory["Coffee"] and has_predicate == HAS_COFFEE)):
                    return [('deliver', has_predicate, obj)]
        return False
    
    deliver_dynamic_method.__name__ = f"deliver2_{has_predicate}_{deliver_predicate}"
    return deliver_dynamic_method 


def add_visitOrPickup(visit_predicate):
    def visitOrPickup_dynamic_method(state, obj):
        if obj == visit_predicate:
            return [('visitOrPickup', visit_predicate)]
        return False

    visitOrPickup_dynamic_method.__name__ = f"visitOrPickup{visit_predicate}"
    return visitOrPickup_dynamic_method

def define_dynamic_methods():
    """
    The method to define the dynamic methods for the office world environment
    """
    dynamic_methods = []
    for obj in [VISITED_A, VISITED_B, VISITED_C, VISITED_D, HAS_MAIL, HAS_COFFEE, VISITED_OFFICE]:
        dynamic_methods.append(add_visitOrPickup(obj))
    for obj in [(HAS_COFFEE, DELIVERED_COFFEE), (HAS_MAIL, DELIVERED_MAIL) ]:
        dynamic_methods.append(add_deliver_2(obj[0], obj[1]))
        dynamic_methods.append(add_deliver(obj[0], obj[1]))
    hop.declare_methods("solve", *dynamic_methods)


def declare_methods_and_operators():
    """
    The method to declare the methods and operators for the office world environment
    """
    # We would declare the methods for the office world environment
    hop.declare_methods("achieve_goal", achieve_goal)


    # We would define the dynamic methods for the office world environment
    define_dynamic_methods()

    # We would declare the operators for the office world environment
    hop.declare_operators(visitOrPickup, deliver)

def get_environment_state(obs, goals):
    """
    The method to get the environment state
    :param obs: The observation from the environment
    :param goals: The goals for the environment
    :return: The current state and the goal state of the office world environment
    """
    # We get the number of agents from the observation
    num_agents = len(obs.keys())

    # We would define the state
    state = hop.State("state")

    # The state would have the location and inventory for each agent indexed by the agent id
    state.agent_locations = {}
    state.agent_inventory = {}

    # The remaining variables define the different taskw we are concerned with
    state.tasks_completed = {}
    tasks = []

    # We would iterate over the observation for the agents to get the state
    for agent_id, agent_obs in obs.items():
        # We would get the agent location and inventory
        state.agent_locations[agent_id] = agent_obs[:2]
        state.agent_inventory[agent_id] = {  "Mail": agent_obs[2*num_agents], "Coffee": agent_obs[2*num_agents+1]}
        # We would get the task for the agent
        if not tasks:
            tasks = list(agent_obs[-9:])

    state.tasks_completed = set([i for i, task in enumerate(tasks) if task])
    
    # We need to set the goal
    goal = hop.State("goal")
    goal.tasks = []
    for i in goals:
        if i not in state.tasks_completed:
            goal.tasks.append(i)
   
    # We would return the state and the goal
    return state, goal


class MultiAgentOfficeWorldPlanner(Planner):
    def __init__(self, obs, target, target_binary=False):
        """
        The constructor for the office world planner
        :param obs: The observation from the environment
        """
        self.obs = obs
       
        self.original_obs_size =list(obs.values())[0].size
        self.plan = None
        self.target_binary = target_binary
        self.target = target
        self.initialize_goals()
        assert type(self.obs) == dict

        # The size of the observation space will tell us the number of agents
        self.agents = list(self.obs.keys())
        num_agents = len(self.agents)


        if num_agents < 1:
            raise ValueError("There are no agents to plan for")
        else:
            # We need to declare the methods and the operators for the domain
            declare_methods_and_operators()
            self.operators = hop.operators.keys()

    
    def initialize_goals(self):
        self.goal = []
        for i in self.target:
            self.goal.append(goal_dict[i])
            
    def reset(self, obs):
        """
        The method to reset the planner (Performs the same role as the constructor
        without needing to reinitialize the object)
        :param obs: The observation from the environment
        :return: None
        """
        self.obs = obs
        self.plan = None
        self.initialize_goals()

        assert type(self.obs) == dict
        self.agents = list(self.obs.keys())

        if len(self.agents) < 1:
            sys.exit("There are no agents to plan for")
        else:
            declare_methods_and_operators()
            self.operators = hop.operators.keys()


    def next_tasks(self, obs, **kwargs):
        """
        The method to get the next tasks for the agents
        :param obs: The observation of the environment
        :param kwargs: The additional arguments to be passed to the method
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
        The method to check if the task is done for the current agent
        :param obs: The current observation
        :param tasks: The tasks for the agents

        :return:  A dictionary indexed by each agent indicating whether the agent has finished a task or not
        The dictionary would have the task if the task is done, False if the task is not done or None if the task is
        None
        """
        tasks_done = dict()
        num_agents = len(self.agents)

        
        # We would need to iterate over all the tasks and the agents
        for agent in self.agents:
            if agent in tasks.keys():
                task = tasks[agent]
                # We would get the index of the agent, which is the numeric value of the agent
                i = int(agent.split("AGENT")[1])
                if task:
                    # If the current task is pickup
                    if task[0] == "visitOrPickup":

                        if np.abs(obs[agent][4*num_agents + task[1]] - 1) < 0.0001:
                            tasks_done[agent] = task[0]
                        else:
                            tasks_done[agent] = False
                    # If the current task is deliver

                    if task[0] == "deliver":
                        # If both the object and the deliver object were performed by
                        # the correct agent
                        if np.abs(obs[agent][4*num_agents + task[2]] - 1) < 0.0001:
                            tasks_done[agent] = task[0]
                        else:
                            tasks_done[agent] = False
                    
                else:
                    tasks_done[agent] = None

        return tasks_done

    def is_plan_valid(self, obs, tasks):
        """
        The method to decide whether the current plan is valid
        """
        num_agents = len(self.agents)
        disrupted_agents = []
        for agent in self.agents:
            if agent in tasks.keys():
                task = tasks[agent]
                # We would get the index of the agent, which is the numeric value of the agent
                i = int(agent.split("AGENT")[1])
           
                
                if task:
                    if task[0] == "visitOrPickup":
             
                            if np.abs(obs[agent][4*num_agents + task[1]] - 0) > 0.0001  and \
                                np.abs(obs[agent][4*num_agents + task[1]] - 1) > 0.0001:
                            
                                disrupted_agents.append(agent)
                   

                    elif task[0] == "deliver":
                        if np.abs(obs[agent][4*num_agents + task[2]] - 0 ) > 0.0001 and \
                                np.abs(obs[agent][4*num_agents + task[2]] - 1) > 0.0001:
                            disrupted_agents.append(agent)

        if len(disrupted_agents)>0:
            return [False, disrupted_agents]
        return [True, []]

    def get_abstract_obs(self, obs, tasks):
        """
        The method to get the abstract observation for the agents based on the current task using the hand written
        D-FOCI rules
        :param obs: The current observation from the environment
        :param tasks: The current tasks assigned to the agent
        """

        # The default abstract observation is a vector of zeros
        # abstract_obs = {agent: np.zeros(2*len(self.agents) + 3) for agent in self.agents}
        abstract_obs = dict()
        # Get the number of agents
        num_agents = len(self.agents)

        # We would iterate over the tasks for the agents
        for agent in tasks.keys():
            task = tasks[agent]

            # If the agent has a task assigned to it, we would get the task specific abstract observation
            if task:
                # Get the object concerning the task
                task_to_be_completed = task[1]

                # We would return the x, y for both the agents, inventory for the current agent, and the task for the
                # current agent
                if task[0] == "visitOrPickup":
                    if(not self.target_binary):
                        abstract_obs[agent] = np.concatenate((obs[agent][:2*num_agents],
                                                        obs[agent][2*num_agents:2*num_agents+2],
                                                        np.expand_dims(obs[agent][4*num_agents + task_to_be_completed],
                                                                        axis=0), np.array([task[1]/10])), axis=0)
                    else:
                        binary_task = format(task[1],'03b')
                        task_list = [int(char) for char in binary_task]
                        # print(task_list)
                        abstract_obs[agent] = np.concatenate((obs[agent][:2*num_agents],
                                    obs[agent][2*num_agents:2*num_agents+2],
                                    np.expand_dims(obs[agent][4*num_agents + task_to_be_completed],
                                                    axis=0), np.array(task_list)), axis=0)

                if task[0] == "deliver":
                    if(not self.target_binary):
                        abstract_obs[agent] = np.concatenate((obs[agent][:2*num_agents],
                                                        obs[agent][2*num_agents:2*num_agents+2],
                                                        np.expand_dims(obs[agent][4*num_agents + task_to_be_completed],
                                                                        axis=0), np.array([task[2]/10])), axis=0)
                        
                    else:
                        binary_task = format(task[2]-3,'03b')
                        task_list = [int(char) for char in binary_task]
                        
                        abstract_obs[agent] = np.concatenate((obs[agent][:2*num_agents],
                                                        obs[agent][2*num_agents:2*num_agents+2],
                                                        np.expand_dims(obs[agent][4*num_agents + task_to_be_completed],
                                                                        axis=0), np.array(task_list)), axis=0)
            else:
                if(not self.target_binary):
                    abstract_obs[agent] = np.concatenate((obs[agent][:2*num_agents],
                                                        np.zeros(4)), axis=0)
                else:
                    abstract_obs[agent] = np.concatenate((obs[agent][:2*num_agents],
                                                      np.zeros(6)), axis=0)

        return abstract_obs

    def get_observation_space_dict(self, obs_dict, mode="abstract"):
        """
        The method returns the observation dictionary for the different agents
        """
        if mode == "abstract":
            if self.target_binary:
                for agent in self.agents:
                    obs_dict[agent] = Box(low=0, high=1, shape=(2*len(self.agents) + 6,), dtype=np.float32)
            else:
                for agent in self.agents:
                    obs_dict[agent] = Box(low=0, high=1, shape=(2*len(self.agents) + 4,), dtype=np.float32)
        elif mode == "extended":
            for agent in self.agents:
                obs_dict[agent] = Box(low=-1, high=1, shape=(self.original_obs_size + 4,), dtype=np.float32)
        else:
            obs_dict = obs_dict

        return obs_dict

    def get_action_space_dict(self, action_dict):
        """
        The method returns the action space for the agents
        :param action_dict: The action space for the agents
        :return: The action space for the agents
        """
        return action_dict

    def get_current_plan(self):
        """
        The method to get the current plan
        :return: The current plan
        """
        return self.plan

    def set_goal(self, goal):
        """
        The method to set the goal for the planner
        :param goal: The goal for the planner
        :return: None
        """
        self.goal = goal


    def get_agents(self):
        """
        The method to get the agents for the planner
        :return: The agents for the planner
        """
        return self.agents

    def get_operators(self):
        """
        The method to get the operators for the planner
        :return: The operators for the planner
        """
        return self.operators

    def get_plan(self, obs):
        """
        The method to get the plan for the agents
        :param obs: The current observation
        :return: The plan for the agents
        """
        # If the observation value for the goal is not 0, we remove the goal from the list of goals
        # print("Hi")
        self.goal = [goal for goal in self.goal if np.abs(obs[self.agents[0]][4*len(self.agents) + goal] - 0) < 0.0001]
        state, goal = get_environment_state(obs, self.goal)
        # print("Current goal", self.goal)!
        # If there are no more goals remaining, we would return None
        if len(goal.tasks) == 0:
            return None

        # We would compute the plan using pyhop
        plan = hop.pyhop(state, [('achieve_goal', goal)], verbose=0)
        # print("Non Distributed Plan", plan)
        # Once, we have the plan, we would divide the plan between the different agents
        if not plan:
            # print(state, goal)
            # print(obs)
            sys.exit("No plan found. Try again")
        
        # We would aggregate the tasks between the different agents
        tasks = aggregate_tasks(plan, aggregator=office_task_aggregator)
        # print("Aggregated Tasks: ", tasks)
        task_duration_dict = calculate_task_duration(tasks)
        # print("Task Duration: ", task_duration_dict)
        # We would calculate the distributed tasks
        distributed_plans = create_distributed_plans(task_duration_dict, len(self.agents))
        # print("Distributed Plans: ", distributed_plans)
        # We need to be careful in how we assign the plans.
        # The plans need to be consistent with the inventory
        agent_assigned_plans = {agent: [] for agent in self.agents}
        agent_object_tuples = []
        object_plan_tuples = set()
        
        # We would append the agent object tuples to the agents
        for agent in self.agents:
            for key, value in state.agent_inventory[agent].items():
                if value:
                    tuple = (agent, 4) if key == "Mail" else (agent, 5)
                    agent_object_tuples.append(tuple)

        # print("Agent Object Tuples: ", agent_object_tuples)
        # We would consider the plans which have the mail and coffee objects
        for plan_no, plan_d in distributed_plans.items():
            for task in plan_d:
                object_plan_tuples.add((task[1], plan_no))
        
        object_plan_tuples = list(object_plan_tuples)
        # print("Object Plan Tuples: ", object_plan_tuples)

        if len(agent_object_tuples) == 0:
            # If there are no restrictions on how the plans are assigned to the agents, we assign the plans 
            # in order
            for i, agent in enumerate(self.agents):
                if i in distributed_plans.keys():
                    agent_assigned_plans[agent] = distributed_plans[i]
                    del distributed_plans[i]
                else:
                    agent_assigned_plans[agent] = []
        
        else:
            # We would assign the plans based on the agents inventory
            plan_assigned = [False for _ in distributed_plans.keys()]
            for agent, obj in agent_object_tuples:
                for plan_no, plan_desc in distributed_plans.items():
                    if (obj, plan_no) in object_plan_tuples:
                        if not plan_assigned[plan_no]:
                            agent_assigned_plans[agent] = plan_desc
                            # print(f"Agent {agent} assigned plan {plan_desc}")
                            plan_assigned[plan_no] = True
                            distributed_plans[plan_no] = None
        
        # Get the keys for the plans not yet assigned
        remaining_plan_keys = [key for key in distributed_plans.keys() if distributed_plans[key] is not None]
        # We have assigned all the agents which mapped to a specific plan
        # We would now assign the remaining plans to the agents
        # print("Remaining Plan Keys: ", remaining_plan_keys )
        for agent in self.agents:
            if not agent_assigned_plans[agent]:
                if remaining_plan_keys:
                    agent_assigned_plans[agent] = distributed_plans[remaining_plan_keys[0]]
                    del remaining_plan_keys[0]
                else:
                    agent_assigned_plans[agent] = []

        # print("Final Agent Assigned Plans: ", agent_assigned_plans)
        
        # The planner should work now. It is still possible that the planner may break. This domain doesn't have the
        # restriction of an agent not being able to deliver both the coffee and the mail at the same time
        # print("Agent Plans: ", agent_assigned_plans)

        return agent_assigned_plans