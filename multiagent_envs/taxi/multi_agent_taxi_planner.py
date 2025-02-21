import sys
import numpy as np

from gym.spaces import Box
from utils import pyhop as hop
from utils.planner import Planner
from utils.ma_plan_scheduler import aggregate_tasks, calculate_task_duration, create_distributed_plans
from .taxi_domain_methods import taxi_task_aggregator


def achieve_goal(state, goals):
    """
    We would check if the goal state has been reached. For us the goal state is
    when all the passengers have been dropped off at their destinations
    """
    goal_achieved = True
    unachieved_goals = []
    for passenger in goals.at_dest:
        # If there is a passenger which has not yet been dropped
        if passenger not in state.dropped:
            goal_achieved = False
            unachieved_goals.append(passenger)
    if goal_achieved:
        return []

    # We have an edge case when the goals are ordered in a way which makes them impossible to accomplish
    # Such a case happens when both the taxis are holding a passenger, and we are trying to transport
    # a different passenger
    # We iterate over the list, moving the passengers which are not in the taxi to the end of the list
    for key, val in state.in_taxi.items():
        if val == 0:
            if key in unachieved_goals:
                unachieved_goals.remove(key)
                unachieved_goals.append(key)
    return [('transport', unachieved_goals[0]), ('achieve_goal', goals)]


# If the passenger is not in the taxi and at the given location, we pick up the passenger
def pickup(state, passenger):
    """
    If the passenger is not in the taxi and at the given location, we pick up the passenger
    :param state: The current state
    :param passenger: The passenger we are trying to pickup

    :returns:  Returns the updated state if we are able to perform the action, else returns False

    """
    # If the passenger is not in any taxi
    if state.in_taxi[passenger] == 0:
        location = state.at[passenger]

        # We assign the passenger to any empty taxi which is available
        for taxi in state.taxi_at.keys():
            empty_taxi = True
            for p, t in state.in_taxi.items():
                if t == taxi:
                    empty_taxi = False
                    break

            if empty_taxi:
                state.in_taxi[passenger] = taxi
                state.taxi_at[taxi] = location
                del state.at[passenger]
                return state
    return False


# If the passenger is in the taxi, then we drop the passenger
def drop(state, passenger):
    """

    :param state: The current state
    :param passenger: The passenger we want to drop off

    :returns: The updated state if we are able to perform the action, else returns False

    """
    for taxi, loc in state.taxi_at.items():
        if state.in_taxi[passenger] == taxi and passenger not in state.dropped:
            location = state.dest[passenger]
            state.in_taxi[passenger] = 0
            state.dropped.append(passenger)
            state.taxi_at[taxi] = location
            return state

    return False


# The add_transport_1 and add_transport_2 methods are used to create dynamic methods for passengers not in the taxi
# or already dropped
def add_transport_1(p):
    def transport_dynamic_method(state, passenger=p):
        if state.in_taxi[passenger] == 0 or passenger not in state.dropped:
            return [('pickup', passenger), ('drop', passenger)]
        return False

    transport_dynamic_method.__name__ = "transport_%s" % (str(p))
    return transport_dynamic_method


# The add transport_2 method is used to create the dynamic methods for passengers already in the taxi
def add_transport_2(p):
    def transport_dynamic_method(state, passenger=p):
        if state.in_taxi[passenger] != 0:
            return [('drop', passenger)]

        return False

    transport_dynamic_method.__name__ = "transport_%s_intaxi" % (str(p))
    return transport_dynamic_method


# The driver method which calls the add_transport_1 and add_transport_2 methods for each passenger to create the
# dynamic methods
def define_dynamic_methods(num_passengers):
    dynamic_methods_1 = []
    dynamic_methods_2 = []
    for passenger in range(1, num_passengers + 1):
        dynamic_methods_1.append(add_transport_1(passenger))
        dynamic_methods_2.append(add_transport_2(passenger))
    dynamic_methods = dynamic_methods_2 + dynamic_methods_1
    hop.declare_methods('transport', *dynamic_methods)


# We create the declare_methods_and_operators method which would create the methods and operators for the taxi domain
def declare_methods_and_operators(num_passengers):
    hop.declare_methods('achieve_goal', achieve_goal)
    define_dynamic_methods(num_passengers)
    hop.declare_operators(pickup, drop)


# The taxi planner class which would create the plans for the taxi domain
def get_environment_state(obs, num_passengers):
    """
    The method for getting the environment state for the taxi domain
    :param obs: The current observation dictionary
    :param num_passengers: The current number of passengers

    :returns: The environment state for the current observation dictionary

    """
    # The observation obs is a vector which is the observation for a single agent
    state = hop.State('state')

    # The taxi_at would store the location for each taxi
    state.taxi_at = dict()

    # We would calculate the number of taxis
    num_taxis = len(obs.keys())

    # We would get the observation for a single agent
    sample_obs = list(obs.values())[0]

    # We would get the locations of the taxi's by reading the num_taxis*2 elements from the observation space
    # We need to  multiply by 10 because the taxi location are given in decimal
    taxi_locations = (10 * sample_obs[0:num_taxis * 2].reshape(num_taxis, 2)).astype(int)

    for i in range(1, len(taxi_locations) + 1):
        state.taxi_at[i] = -1

    # We can use all the calculations from before to construct vectors for each passenger
    passenger_vec = sample_obs[num_taxis * 2:].reshape(num_passengers, 9)

    # We can basically use the passenger_vec to calculate which taxi contains which passengers
    passenger_pick_and_drop = np.argwhere(passenger_vec == 1.0)

    # The state would also contain the current and destination location for each passenger
    state.at, state.dest = {}, {}

    # We would iterate over the passenger_pick_and_drop to get the pickup and drop locations for each passenger
    for passenger_index, bin_index in passenger_pick_and_drop:
        is_pickup, is_drop = bin_index < 4, bin_index > 4
        if is_pickup:
            if passenger_vec[passenger_index, 4] == 0:
                state.at[passenger_index + 1] = bin_index
        if is_drop:
            state.dest[passenger_index + 1] = bin_index - 5

    # The middle element of each row tells us the current pickup and drop status of the passenger
    state.in_taxi = list(passenger_vec[:, 4])

    # We would convert the state.in_taxi to dictionary
    state.in_taxi = {i + 1: int(10 * state.in_taxi[i]) for i in range(len(state.in_taxi))}

    # We need to also store all the passengers that have been dropped
    state.dropped = []

    # If the passenger doesn't have a pickup or a drop location, then we assume that the passenger has been dropped
    for passenger in range(1, num_passengers + 1):
        if passenger not in state.at.keys() and passenger not in state.dest.keys():
            state.dropped.append(passenger)
    # We need to also set the goal state
    goal = hop.State("goal")
    goal.at_dest = list(state.at.keys())
    goal.at_dest = goal.at_dest + [i for i in state.dest.keys() if i not in state.dropped]
    goal.at_dest = list(set(goal.at_dest))

    # We would return the state and the goal
    return state, goal


class MultiAgentTaxiPlanner(Planner):
    def __init__(self, obs):
        """
        The init method for the taxi planner
        :param obs: The observation received by the planner
        """
        self.obs = obs
        self.original_obs_size = list(self.obs.values())[0].size
        self.plan = None
        self.goal = None

        assert type(self.obs) == dict

        # The observation space will tell us the number of agents
        self.agents = list(self.obs.keys())
        num_taxis = len(self.agents)

        # The planner would exit with an error if there are no agents
        if num_taxis < 1:
            sys.exit("There are no agents to plan for")
        else:
            # We need to calculate the number of passengers
            obs_size = list(self.obs.values())[0].size
            self.num_passengers = int((obs_size - num_taxis * 2) / 9)
            declare_methods_and_operators(self.num_passengers)
            self.operators = hop.operators.keys()

    def reset(self, obs):
        """
        The reset method for the taxi planner
        :param obs: The current observation state

        :returns: No return. Performs the same role as the constructor, resetting the setting based on the observation.
                  Currently, the layout is not changed.

        """
        self.obs = obs
        self.plan = None
        self.goal = None

        assert type(self.obs) == dict

        # The observation space will tell us the number of agents
        self.agents = list(self.obs.keys())
        num_taxis = len(self.agents)

        # The planner would exit with an error if there are no agents
        if num_taxis < 1:
            sys.exit("There are no agents to plan for")
        else:

            # We need to calculate the number of passengers
            obs_size = list(self.obs.values())[0].size
            self.num_passengers = int((obs_size - num_taxis * 2) / 9)
            declare_methods_and_operators(self.num_passengers)
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
        Checks if the task is done for the agents

        :param obs: The current observation dictionary for agents
        :param tasks: The dictionary of tasks for each agent

        :returns: A dictionary indexed by each agent indicating whether the agent has finished a task or not
        The dictionary would have the task if the task is done, False if the task is not done or None if the task is
        None
        """

        # We need to get the number of agents
        num_agents = len(self.agents)
        tasks_done = dict()
        # We would iterate over all the agents and tasks
        for agent, agent_obs in obs.items():
            if agent in tasks.keys():
                task = tasks[agent]
                current_agent_index = self.agents.index(agent)

                if task:
                    # We get the current task type as well as the passenger number
                    task_type, passenger_id = task
                    # If the task is pickup, we need to check if the passenger is in the taxi
                    if task_type == "pickup":
                        # If the passenger is in the correct taxi, it has been picked up
                        if agent_obs[2 * num_agents + (passenger_id - 1) * 9 + 4] == (current_agent_index+1)/10:
                            tasks_done[agent] = task_type
                        else:
                            tasks_done[agent] = False

                    if task_type == "drop":
                        # If the task is drop, we would check the drop location for the passenger. If all the possible drop
                        # locations are 0, the particular task is done
                        if np.array_equal(np.array([0, 0, 0, 0]),
                                          agent_obs[2 * num_agents + (passenger_id - 1) * 9 + 5:
                                          2 * num_agents + (passenger_id - 1) * 9 + 9]):
                            tasks_done[agent] = task_type

                        else:
                            tasks_done[agent] = False
                else:
                    tasks_done[agent] = None

        return tasks_done

    def is_plan_valid(self, obs, tasks):
        """
        The method to check if the current plan is valid
        """

        #must add disrupted agents list 
        
        # We need to get the number of agents
        num_agents = len(self.agents)
        disrupted_agents = []
        # We would iterate over all the agents and tasks
        for agent, agent_obs in obs.items():
            # If the agent has been assigned a pickup which has been done by some other agent
            if agent in tasks.keys():
                task = tasks[agent]
                current_agent_index = self.agents.index(agent)

                if task:
                    # We get the current task type as well as the passenger number
                    task_type, passenger_id = task
                    # If the task is pickup, we need to check if the passenger is in the taxi
                    if task_type == "pickup":
                        # If the passenger is not in the correct taxi, it has been picked up
                        if (agent_obs[2 * num_agents + (passenger_id - 1) * 9 + 4] != 0 and
                                agent_obs[2 * num_agents + (passenger_id - 1) * 9 + 4] != (current_agent_index+1)/10):
                            disrupted_agents.append(agent)

                    if task_type == "drop":
                        # If the task is drop, we would check whether the passenger is in the correct taxi, or it has
                        # aleady been dropped
                        if (agent_obs[2 * num_agents + (passenger_id - 1) * 9 + 4] != (current_agent_index+1)/10) and \
                                not np.array_equal(np.array([0, 0, 0, 0]),
                                agent_obs[2 * num_agents + (passenger_id - 1) * 9 + 5:
                                          2 * num_agents + (passenger_id - 1) * 9 + 9]):
                            disrupted_agents.append(agent)

        return len(disrupted_agents) == 0, disrupted_agents

    def get_abstract_obs(self, obs, tasks):
        # For the abstract observation, regardless of the pickup or drop task, we
        # will output the taxis as well as the

        # Get the passenger involved in the current task

        num_agents = len(self.agents)

        # The observation I am currently returning is for a single agent

        # We would concatenate all the taxi locations as well as the 9 tuple for the agent in question
        # The indexing for passenger_id is different from passenger itself. It is 1 over the actual index
        abstract_obs = dict()

        for agent, agent_obs in obs.items():
            if agent in tasks.keys():
                task = tasks[agent]
                current_agent_index = self.agents.index(agent)

                if task:
                    task_type, passenger_id = task
                    # We would get the passenger id from the task
                    if task_type == "pickup":
                        abstract_obs[agent] = np.concatenate((agent_obs[2*current_agent_index:2*current_agent_index+2],
                                                              agent_obs[:2 * current_agent_index],
                                                              agent_obs[2 * current_agent_index + 2:2*num_agents],
                                                              agent_obs[2 * num_agents + (passenger_id - 1) * 9:
                                                                        2 * num_agents + (passenger_id - 1) * 9 + 5]))
                    if task_type == "drop":
                        abstract_obs[agent] = np.concatenate((agent_obs[2*current_agent_index:2*current_agent_index+2],
                                                              agent_obs[:2 * current_agent_index],
                                                              agent_obs[2 * current_agent_index + 2:2*num_agents],
                                                              agent_obs[2 * num_agents + (passenger_id - 1) * 9 + 5:
                                                                        2 * num_agents + (passenger_id - 1) * 9 + 9],
                                                              agent_obs[2 * num_agents + (passenger_id - 1) * 9 + 4:
                                                                        2 * num_agents + (passenger_id - 1) * 9 + 5]
                                                              ))
                else:
                    abstract_obs[agent] = np.concatenate((agent_obs[:2 * num_agents], np.zeros(5)))

        return abstract_obs

    def get_observation_space_dict(self, obs_dict, mode="abstract"):
        """
        Gets the observation space for the agent
        :param obs_dict: The observation dictionary for the agents
        :param mode: The mode for the observation space
        :returns: The abstract observation dictionary for the agents
        """
        for agent in self.agents:
            if mode == "abstract":
                obs_dict[agent] = Box(low=0, high=1, shape=(2 * len(self.agents) + 5,), dtype=np.float32)
            elif mode == "extended":
                obs_dict[agent] = Box(low=-1, high=1, shape=(self.original_obs_size + 4, ), dtype=np.float32)
            else:
                continue

        return obs_dict

    def get_action_space_dict(self, action_dict):
        """
        Gets the action space for the agent
        :param action_dict:  The action dictionary received from the environment

        :returns: The action dictionary for the agent

        """
        return action_dict

    def set_goal(self, _goal):
        """
        Sets the goal for the agent
        :param _goal: The goal for the agent
        """
        self.goal = _goal

    def get_agents(self):
        """
        Gets the agents for the planner
        :returns: The list of agents for the current planner
        """
        return self.agents

    def get_operators(self):
        """
        Returns the operators for the current planner
        :returns: The list of operators for the current planner
        """
        return self.operators

    def get_plan(self, obs):
        """
        Gets the plan for the agents
        :obs: Gets the current observation dictionary for the agent
        :returns: Returns a dictionary of plans for each agent
        """
        # This is where we would be using Py-hop to generate the main plan
        state, goal = get_environment_state(obs, self.num_passengers)

        # If there are no goals remaining, we would return None
        if len(goal.at_dest) == 0:
            return None

        # Get the common plan using Pyhop
        plan = hop.pyhop(state, [('achieve_goal', goal)])

        # Call the scheduler to divide the plans for all the agents
        if not plan:
            print(state,goal)
            print(obs)
            sys.exit("No plan found. Try again")
        tasks = aggregate_tasks(plan, aggregator=taxi_task_aggregator)
        task_duration_dict = calculate_task_duration(tasks)

        # We would calculate the distributed plans
        distributed_plans = create_distributed_plans(task_duration_dict, num_agents=len(self.agents))

        # We need to be careful when we assign these plans to agents in case when a passenger is in a taxi
        # We can't assign the agent with a passenger on board, a plan without the passenger

        # We need to parse over the current observation to get which agent has a passenger on board
        # We would then assign the appropriate plan to the agent

        agent_assigned_plans = dict()

        # We would store if there are any agents which have a passenger on board
        # We would also store which passengers are involved in any given plan

        agent_passenger_tuples = []
        passenger_plan_tuples = set()


        for p in range(1, self.num_passengers+1):
            if state.in_taxi[p] != 0:
                agent_passenger_tuples.append((p, state.in_taxi[p] - 1))

        for plan_no, plan_d in distributed_plans.items():
            for task in plan_d:
                passenger_plan_tuples.add((task[1], plan_no))

        passenger_plan_tuples = list(passenger_plan_tuples)

        # # In case there is a discrepancy between the passenger and plan tuple, we move the passenger tasks to
        # # the agent with the passenger on board
        # temp = []
        # for passenger, plan_p in passenger_plan_tuples:
        #     append_flag = False
        #     for plan_a in distributed_plans.keys():
        #         # If there is a discrepancy between the passenger and plan tuple, we move the passenger tasks to
        #         # the plan for the taxi with the passenger on board
        #         if (passenger, plan_a) in agent_passenger_tuples and plan_a != plan_p:
        #             for task in distributed_plans[plan_p]:
        #                 if task[1] == passenger:
        #                     distributed_plans[plan_a].append(task)
        #                     distributed_plans[plan_p].remove(task)
        #                     temp.append((passenger, plan_a))
        #                     append_flag = True
        #
        #     if not append_flag:
        #         temp.append((passenger, plan_p))
        #
        # # Update the passenger_plan_tuples to reflect the changes made regarding the plan containing the passenger
        # passenger_plan_tuples = temp

        # If there are no agents with a passenger on board, we would assign the plans to the agents
        if len(agent_passenger_tuples) == 0:
            for i, agent in enumerate(self.agents):
                if i in distributed_plans.keys():
                    agent_assigned_plans[agent] = distributed_plans[i]
                    del distributed_plans[i]
                else:
                    agent_assigned_plans[agent] = []
        else:
            # We get the remaining plans left and assign them a False
            flag_plans = [False for _ in distributed_plans.keys()]
            for passenger, agent in agent_passenger_tuples:
                for plan_no, plan_d in distributed_plans.items():
                    if (passenger, plan_no) in passenger_plan_tuples:
                        if not flag_plans[plan_no]:
                            agent_assigned_plans[self.agents[agent]] = plan_d
                            flag_plans[plan_no] = True
                            distributed_plans[plan_no] = None

        # Get the keys for the plans not yet assigned
        remaining_plan_keys = [key for key in distributed_plans.keys() if distributed_plans[key] is not None]
        # We have assigned all the agents which mapped to a specific plan
        # We would now assign the remaining plans to the agents

        for agent in self.agents:
            if agent not in agent_assigned_plans.keys():
                if remaining_plan_keys:
                    agent_assigned_plans[agent] = distributed_plans[remaining_plan_keys[0]]
                    del remaining_plan_keys[0]
                else:
                    agent_assigned_plans[agent] = []

        # It is still possible that we may have passengers in the taxi which are not in the plan
        for i, agent in enumerate(self.agents):
            plan = agent_assigned_plans[agent]
            if plan:
                for task in plan:
                    # In case, we have a passenger who is in the plan for the taxi but present in a different taxi
                    if task[0] == "drop" and state.in_taxi[task[1]] != i+1 and state.in_taxi[task[1]] != 0:
                        agent_assigned_plans[agent].remove(task)

                        # We need to find the taxi which has the passenger and assign the task to that taxi
                        agent_assigned_plans[self.agents[state.in_taxi[task[1]]-1]].append(task)

        # Before we go over the plan, we would need to make sure that any drop is preceded by a pickup for the same
        # passenger, unless there is no pickup for that passenger
        for agent in self.agents:
            plan_a = agent_assigned_plans[agent]
            if plan_a:
                for i, task in enumerate(plan_a):
                    # If the current task is drop, but there is no pickup for the same passenger in the previous line
                    if i != 0 and task[0] == "drop" and plan_a[i - 1][0] != "pickup" and task[1] != plan_a[i - 1][1]:
                        plan_a.remove(task)
                        plan_a.insert(0, task)
                        break


        return agent_assigned_plans
