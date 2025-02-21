import copy
import sys
import numpy as np
from utils.ma_plan_scheduler import aggregate_tasks, calculate_task_duration, create_distributed_plans
from multiagent_envs.collect_and_build.collect_and_build_methods import collect_and_build_task_aggregator
from multiagent_envs.collect_and_build.env_descriptors import *
from gym.spaces import Box
from utils.planner import Planner
from utils import pyhop as hop

# The current planner works on the assumption that there are exactly 4 resource tiles in the environment
# The number of agents and the buildings can vary.
# It would not work if the number of resource tiles is not 4

def achieve_goal(state, goal):
    """
    The method to check whether the goal has been achieved
    :param state: The current state of the environment
    :param goal: The goals to achieve
    """
    goals_achieved = True
    unfulfilled_goals = []
    for b_id in goal.to_build:
        building = f"building_{b_id}"
        # If the building has not been built, we would add it to the unfulfilled goals
        if np.abs(state.buildings[building]["built_status"] - 0) < 1e-4:
            unfulfilled_goals.append(b_id)
            goals_achieved = False

    if goals_achieved:
        return []

    return [("construct_building", unfulfilled_goals[0]), ("achieve_goal", goal)]

# The operators which we need to define are collect and build

def collect(state, resource_type, building_for_resource):
    """
    The operator to collect the following resource
    :param state: The current state of the environment
    :param resource_type: The resource to be collected
    :param building_for_resource: The building for which the resource is to be collected
    """
    # When we see division by 100, it is because the observation outputs the resources in the form of x/100
    resource_required = None
    # We would check if we already have the resources to build the building
    building_for_resource = f"building_{building_for_resource}"
    if state.buildings[building_for_resource]["type"] == "house":
        resource_required = house["cost"]
    if state.buildings[building_for_resource]["type"] == "mansion":
        resource_required = mansion["cost"]
    if state.buildings[building_for_resource]["type"] == "castle":
        resource_required = castle["cost"]

    resource_required = {resource: resource_required[resource]/100 for resource in resource_required.keys()}

    # We would need to consider the resources we have already used in construction
    resources_used = {key: value for key, value in state.buildings[building_for_resource]["resources_used"].items()}

    resource_required = {resource: resource_required[resource] - resources_used[resource]
                         for resource in resource_required.keys()
                         if resource_required[resource] > resources_used[resource]}

    resources_already_collected = 0
    # We need to check if the resources required to build the building are already available
    for agent in state.agents.keys():
        if resource_type in state.agents[agent]["resources"].keys():
            resources_already_collected += state.agents[agent]["resources"][resource_type]

    # If the resources are already collected, we would return False as the operator cannot be applied
    if resources_already_collected >= resource_required[resource_type]:
        return False

    # If we have reached this point in execution, we need to check how much resources we need to collect
    resources_to_collect = resource_required[resource_type] - resources_already_collected

    # We need to check if the total resources available in the environment are enough to build the building
    total_resources_available = 0
    for resource_tile in state.resources.keys():
        if state.resources[resource_tile]["type"] == resource_type:
            total_resources_available += state.resources[resource_tile]["count"]

    # if the total resource available is less than the resources required, we would return False
    # This should never happen as we would not have reached this point in execution
    if total_resources_available < resources_to_collect:
        return False

    # We would return the state after an agent has collected the desired resource
    max_resource_inventory = MAX_INVENTORY_PER_RESOURCE/100
    for agent in state.agents.keys():
        agent_collecting_resource = False
        agent_resource_inventory_count = state.agents[agent]["resources"][resource_type]
        if agent_resource_inventory_count < max_resource_inventory:
            for resource in state.resources.keys():
                if state.resources[resource]["type"] == resource_type:
                    # We would check if the resource tile has the resource available
                    if state.resources[resource]["count"] > 1e-4:
                        resources_collected_by_agent = min(resources_to_collect,
                                                           max_resource_inventory - agent_resource_inventory_count,
                                                           state.resources[resource]["count"])
                        if resources_collected_by_agent > 1e-4:
                            state.resources[resource]["count"] -= resources_collected_by_agent
                            state.agents[agent]["resources"][resource_type] += resources_collected_by_agent
                            resources_to_collect -= resources_collected_by_agent
                            agent_resource_inventory_count += resources_collected_by_agent
                            agent_collecting_resource = True

        if agent_collecting_resource:
            return state

    # If it is not possible for any agent to collect any amount of the resource, we would return False
    return False

def build(state, building):
    """
    The operator to build the following building
    :param state: The current state of the environment
    :param building: The building we need to build
    """
    building = f"building_{building}"
    max_resource_inventory = MAX_INVENTORY_PER_RESOURCE/100
    # We would check if the building has already been built
    if np.abs(state.buildings[building]["built_status"] - 0) > 1e-4:
        return False

    # We would check if the resources required to build the building are already available
    resources_required = None
    if state.buildings[building]["type"] == "house":
        resources_required = house["cost"]
    if state.buildings[building]["type"] == "mansion":
        resources_required = mansion["cost"]
    if state.buildings[building]["type"] == "castle":
        resources_required = castle["cost"]

    resources_required = {resource: resources_required[resource]/100 for resource in resources_required.keys()}

    # We need to subtract the resources already used in construction from the resources required
    resources_already_used = {key: value for key, value in state.buildings[building]["resources_used"].items()}
    resources_required = {resource: resources_required[resource] - resources_already_used[resource]
                          for resource in resources_required.keys()}

    resources_already_collected = {}
    for agent in state.agents.keys():
        for resource in state.agents[agent]["resources"].keys():
            if resource in resources_required.keys():
                if resource not in resources_already_collected.keys():
                    resources_already_collected[resource] = 0
                resources_already_collected[resource] += state.agents[agent]["resources"][resource]

    # We would check if the resources required to build the building are already available
    resources_available = {r: True for r in resources_required.keys()}
    for resource in resources_required.keys():
        if resources_already_collected[resource] < resources_required[resource]:
            resources_available[resource] = False
            break

    # If all the resources are not available but the agent inventory is full, we would continue with the build
    all_agents_resource_inventory_full = True
    if not resources_available:
        for agent in state.agents.keys():
            for resource in state.agents[agent]["resources"].keys():
                if resources_available[resource] == False \
                        and state.agents[agent]["resources"][resource] < max_resource_inventory:
                    all_agents_resource_inventory_full = False
                    break

    # If the resources are not available and the agent inventory is not full, we would return False
    if not resources_available and not all_agents_resource_inventory_full:
        return False

    # We would now perform the build operation
    build_operation_performed = False
    for agent in state.agents.keys():
        for resource in state.agents[agent]["resources"].keys():
            if resource in resources_required.keys() and resources_required[resource] >  1e-4:
                build_operation_performed = True
                resources_used_by_agent = min(resources_required[resource], state.agents[agent]["resources"][resource])
                resources_required[resource] -= resources_used_by_agent
                state.agents[agent]["resources"][resource] = state.agents[agent]["resources"][resource] \
                                                             - resources_used_by_agent
                state.buildings[building]["resources_used"][resource] += resources_used_by_agent
                state.buildings[building]["agents_involved_in_construction"][agent] = True

    # We would check if the building has been built
    if build_operation_performed:
        if np.abs(sum(resources_required.values()) - 0) < 1e-4:
            state.buildings[building]["built_status"] = 1

        # Since the build operation has been performed, we would return the state
        return state

    # If the build operation has not been performed, we would return False
    return False

def add_collect_and_build_methods(building_id):
    """
    The method to add the collect and build methods to the Pyhop planner
    :param building_id: The id of the building to be built
    """
    def collect_and_build_dynamic_methods(state, b_id = building_id):
        """
        The method to add the collect and build methods to the Pyhop planner
        :param state: The current state of the environment
        :param b_id: The id of the building to be built
        """
        max_resource_inventory = MAX_INVENTORY_PER_RESOURCE/100
        if np.abs(state.buildings[f"building_{b_id}"]["built_status"] - 0) < 1e-4:
            tasks = []
            building_to_build = state.buildings[f"building_{b_id}"]
            resources_required = None
            if building_to_build["type"] == "house":
                resources_required = house["cost"]
            if building_to_build["type"] == "mansion":
                resources_required = mansion["cost"]
            if building_to_build["type"] == "castle":
                resources_required = castle["cost"]

            resources_used = {key: value for key, value in building_to_build["resources_used"].items()}
            resources_required = {resource: resources_required[resource]/100 for resource in resources_required.keys()}
            resources_required = {resource: resources_required[resource] - resources_used[resource]
                                  for resource in resources_required.keys() if resource not in resources_used.keys() or
                                 resources_required[resource] > resources_used[resource]}

            agent_resource_count = {}
            for agent in state.agents.keys():
                agent_resource_count[agent] = {}
                for resource in state.agents[agent]["resources"].keys():
                    agent_resource_count[agent][resource] = state.agents[agent]["resources"][resource]

            # We would check if the resources required to build the building are already available
            resources_already_collected = {}
            for agent in state.agents.keys():
                for resource in state.agents[agent]["resources"].keys():
                    if resource in resources_required.keys():
                        if resource not in resources_already_collected.keys():
                            resources_already_collected[resource] = 0
                        resources_already_collected[resource] += state.agents[agent]["resources"][resource]

            # We would filter the agents based on the resources they have
            resources_required = {resource: resources_required[resource] - resources_already_collected[resource]
                                  for resource in resources_required.keys()
                                  if resources_required[resource] - resources_already_collected[resource] > 1e-4}

            # We need to make sure that we have rounded everything to two decimal places for resources required
            # Otherwise, the number of rounds of collection required would not be calculated correctly
            resources_required = {resource: np.round(resources_required[resource], 2)
                                  for resource in resources_required.keys()}

            num_rounds_of_collection_required = {}
            if len(resources_required) > 0:

                for resource, amount_required in resources_required.items():
                    # We would calculate the number of times we need to collect the resource
                    num_rounds_of_collection_required[resource] = np.ceil(amount_required/max_resource_inventory)

            # We need to handle the resources which are already collected by the agents in case
            # it has resulted in a full inventory for the agents
            # We count the number of agents who already have a full inventory for each resource
            agents_with_full_inventory = {}
            for agent in state.agents.keys():
                agents_with_full_inventory[agent] = {}
                for resource in state.agents[agent]["resources"].keys():
                    agents_with_full_inventory[agent][resource] = False
                    if state.agents[agent]["resources"][resource] == max_resource_inventory:
                        agents_with_full_inventory[agent][resource] = True

            # Calculating agents with full inventory is required to decide when to call the first build task is called
            agent_ordering = sorted(list(state.agents.keys()))
            for agent in agent_ordering:
                for resource in resources_required.keys():
                    # A collection task is called for the first time when the agent does not have a full inventory
                    # for the resource
                    if not agents_with_full_inventory[agent][resource]:
                        if num_rounds_of_collection_required[resource] > 1e-4:
                            tasks.append(("collect", resource, b_id))
                            num_rounds_of_collection_required[resource] -= 1

            # We would add the first build task to the tasks list for each agent which contributes to the build
            if len(tasks) > 0:
                tasks.append(("build", b_id))

            # We would append the remaining collection tasks and build tasks to the tasks list
            # A build task is called every time a collection task is called
            while sum(num_rounds_of_collection_required.values()) > 1e-4:
                for agent in agent_ordering:
                    for resource in resources_required.keys():
                        if num_rounds_of_collection_required[resource] > 1e-4:
                            tasks.append(("collect", resource, b_id))
                            num_rounds_of_collection_required[resource] -= 1
                tasks.append(("build", b_id))

            if tasks:
                return tasks

        return False
    collect_and_build_dynamic_methods.__name__ = f"collect_and_build_dynamic_methods_{building_id}"
    return collect_and_build_dynamic_methods

def add_collect_and_build_methods_2(building_id):
    """
    The method to add the collect and build methods to the Pyhop planner
    :param building_id: The id of the building to be built
    """
    def build_only_dynamic_methods(state, b_id = building_id):
        """
        The method to add the collect and build methods to the Pyhop planner
        :param state: The current state of the environment
        :param b_id: The id of the building to be built
        """
        # We would continue with the build operation if the building has not been built
        if np.abs(state.buildings[f"building_{b_id}"]["built_status"] - 0) < 1e-4:
            # We need to check if we have the resources to build the building
            building_to_build = state.buildings[f"building_{b_id}"]
            resources_required = None
            if building_to_build["type"] == "house":
                resources_required = house["cost"]
            if building_to_build["type"] == "mansion":
                resources_required = mansion["cost"]
            if building_to_build["type"] == "castle":
                resources_required = castle["cost"]

            # We would consider the resources that are still required to build the building
            resources_used = {key: value for key, value in building_to_build["resources_used"].items()}
            resources_required = {resource: resources_required[resource] for resource in resources_required.keys()}
            resources_required = {resource: resources_required[resource]/100 - resources_used[resource]
                                  for resource in resources_required.keys() if resource not in resources_used.keys() or
                                                        resources_required[resource] > resources_used[resource]}

            # We would consider whether the resources required to build the building have already been collected

            resources_already_collected = {}
            for agent in state.agents.keys():
                for resource in state.agents[agent]["resources"].keys():
                    if resource in resources_required.keys():
                        if resource not in resources_already_collected.keys():
                            resources_already_collected[resource] = 0
                        resources_already_collected[resource] += state.agents[agent]["resources"][resource]

            # We would filter the agents based on the resources they have
            resources_required = {resource: resources_required[resource] - resources_already_collected[resource]
                                    for resource in resources_required.keys()
                                  if resources_required[resource] - resources_already_collected[resource] > 1e-4}

            # If we do not need to collect any resources, we would return the build task
            if len(resources_required) == 0:
                return [("build", b_id)]

            # Since, we need to collect resources, we would return False
            return False

        return False
    build_only_dynamic_methods.__name__ = f"build_only_dynamic_methods_{building_id}"
    return build_only_dynamic_methods


def define_dynamic_methods(num_buildings):
    """
    The method to define the dynamic methods for the collect and build environment
    """
    # We would define the dynamic methods for the collect and build environment
    dynamic_methods = []
    for i in range(1, num_buildings+1):
        dynamic_methods.append(add_collect_and_build_methods_2(i))
        dynamic_methods.append(add_collect_and_build_methods(i))

    hop.declare_methods("construct_building", *dynamic_methods)


def declare_methods_and_operators(num_buildings):
    """
    The method to declare the methods and operators for the collect and build environment
    """
    # We would declare the methods for the collect and build environment
    hop.declare_methods("achieve_goal", achieve_goal)
    # We would define the dynamic methods for the collect and build environment
    define_dynamic_methods(num_buildings)
    # We would declare the operators for the collect and build environment
    hop.declare_operators(collect, build)

def get_environment_state(obs):
    """
    The function to get the environment state from the observation
    :param obs: The current observation
    :return: The state corresponding to the observation
    """
    # We get the number of agents from the environment
    agents = list(obs.keys())

    # We would get the number of buildings we need to build from the environment
    num_buildings = int((len(obs[agents[0]]) - 5*len(agents) - 4*RESOURCE_TILES_COUNT)/6)

    # We would define the state in Pyhop
    state = hop.State("state")
    state.agents = {agent: {} for agent in agents}
    state.resources = {}
    state.buildings = {}

    # We would iterate over the observations for the different agents
    for agent in agents:
        agent_observation = obs[agent]
        agent_observation = np.round(agent_observation, 2)

        # We would get the agent's position (x, y), orientation for the agent
        state.agents[agent]["position"] = (agent_observation[0], agent_observation[1])
        state.agents[agent]["orientation"] = agent_observation[2]

        # We would get the resources collected by the agent
        state.agents[agent]["resources"] = {}
        state.agents[agent]["resources"]["wood"] = agent_observation[3]
        state.agents[agent]["resources"]["stone"] = agent_observation[4]

    # Once we have the agents' data, we would get the resources information
    sample_observation = obs[agents[0]]
    sample_observation = np.round(sample_observation, 2)
    for i in range(RESOURCE_TILES_COUNT):
        # If the resource tile exists and not just a placeholder tile
        if np.abs(sample_observation[5*len(agents)+4*i+2] - 0) > 1e-4:
            resource = f"resource_{i+1}"
            state.resources[resource] = {}
            state.resources[resource]["position"] =  sample_observation[5*len(agents)+4*i:5*len(agents)+4*i+2]

            if np.abs(sample_observation[5*len(agents)+4*i+2] - 0.01) < 1e-4:
                state.resources[resource]["type"] = "wood"
            if np.abs(sample_observation[5*len(agents)+4*i+2] - 0.02) < 1e-4:
                state.resources[resource]["type"] = "stone"
            state.resources[resource]["count"] = sample_observation[5*len(agents)+4*i+3]

    # We would get the buildings information
    for j in range(num_buildings):
        # If the building exists and not just a placeholder building
        if np.abs(sample_observation[5*len(agents)+4*RESOURCE_TILES_COUNT+6*j+2] - 0) > 1e-4:
            building = f"building_{j+1}"
            state.buildings[building] = {}
            state.buildings[building]["position"] = sample_observation[5*len(agents)+4*RESOURCE_TILES_COUNT+6*j:
                                                                       5*len(agents)+4*RESOURCE_TILES_COUNT+6*j+2]
            if np.abs(sample_observation[5*len(agents)+4*RESOURCE_TILES_COUNT+6*j+2] - 0.01) < 1e-4:
                state.buildings[building]["type"] = "house"
            if np.abs(sample_observation[5*len(agents)+4*RESOURCE_TILES_COUNT+6*j+2] - 0.02) < 1e-4:
                state.buildings[building]["type"] = "mansion"
            if np.abs(sample_observation[5*len(agents)+4*RESOURCE_TILES_COUNT+6*j+2] - 0.03) < 1e-4:
                state.buildings[building]["type"] = "castle"


            state.buildings[building]["resources_used"] = {}
            state.buildings[building]["resources_used"]["wood"] = sample_observation[5*len(agents)+
                                                                                     4*RESOURCE_TILES_COUNT+6*j+3]
            state.buildings[building]["resources_used"]["stone"] = sample_observation[5*len(agents)+
                                                                                      4*RESOURCE_TILES_COUNT+6*j+4]
            state.buildings[building]["built_status"] = sample_observation[5*len(agents)+4*RESOURCE_TILES_COUNT+6*j+5]
            state.buildings[building]["agents_involved_in_construction"] = {agent: False for agent in agents}

    # We would now define the goal state for the planner
    goal = hop.State("goal")
    goal.buildings_built = []
    goal.to_build = []
    for building in state.buildings:
        b_id = int(building.split("_")[1])
        # If the building is not built, we would add it to the goal state
        if np.abs(state.buildings[building]["built_status"] - 0) < 1e-4:
            goal.to_build.append(b_id)
        else:
            goal.buildings_built.append(b_id)

    return state, goal

class MultiAgentCollectAndBuildPlanner(Planner):
    def __init__(self, obs):
        """
        The planner for the multi-agent collect and build environment
        :param obs: The current observation of the environment
        """
        self.obs = obs
        self.plan = None
        self.goal = None

        assert(type(self.obs) == dict)

        # The observation space will tell us the number of agents
        self.agents = list(self.obs.keys())
        num_agents = len(self.agents)

        # If the number of agents is less than 1, there are no agents to plan for
        if num_agents < 1:
            sys.exit("The number of agents is less than 1")
        else:
            # We need to calculate the number of buildings we need to build
            self.num_buildings = int((len(self.obs[self.agents[0]]) - 5*num_agents - 4*RESOURCE_TILES_COUNT)/6)
            declare_methods_and_operators(self.num_buildings)
            self.operators = hop.operators.keys()

    def get_agents(self):
        """
        The method to get the agents from the observation
        """
        return self.agents

    def reset(self, obs):
        """
        The method to reset the planner with a new observation
        :param obs: The new observation
        """
        self.obs = obs
        self.plan = None
        self.goal = None

        assert (type(self.obs) == dict)

        # The observation space will tell us the number of agents
        self.agents = list(self.obs.keys())
        num_agents = len(self.agents)

        # If the number of agents is less than 1, there are no agents to plan for
        if num_agents < 1:
            sys.exit("The number of agents is less than 1")
        else:
            # We need to calculate the number of buildings we need to build
            self.num_buildings = int((len(self.obs[self.agents[0]]) - 6*num_agents - 4*RESOURCE_TILES_COUNT)/6)
            declare_methods_and_operators(self.num_buildings)
            self.operators = hop.operators.keys()

    def next_tasks(self, obs):
        """
        The method to get the next tasks for the agents
        :param obs: The current observation
        :return: The next tasks for the agents
        """
        next_task = dict()

        self.plan = self.get_plan(obs)

        # Tha planner is a dictionary with a plan for each agent
        if self.plan is None:
            return {}

        # Append the first task for the current plan to the next tasks
        for agent, plan in self.plan.items():
            if plan:
                next_task[agent] = plan[0]
            else:
                next_task[agent] = None

        return next_task

    def get_current_plan(self):
        """
        The method to get the current plan
        :return: The current plan
        """
        return self.plan

    def is_task_done(self, obs, tasks):
        """
        The method to check if the task is done
        :param obs: The current observation
        :param tasks: The tasks to be checked
        :return: The dictionary which indicates whether the task for the agent are done
        """
        tasks_done = {agent: False for agent in self.agents}
        for agent, task in tasks.items():
            if task:
                if task[0] == "collect":
                    # We would get the agent's position (x, y) and orientation to get the location to collect the
                    # resource from
                    resource_type = task[1]
                    building_for_resource = task[2] - 1

                    # We would check if the agent inventory for the resource is full. In case the inventory is full,
                    # the collection task is done, and we would return True
                    max_resource_inventory = np.round(MAX_INVENTORY_PER_RESOURCE/100, 2)
                    if resource_type == "wood":
                        if np.abs(obs[agent][3] - max_resource_inventory) < 1e-4:
                            tasks_done[agent] = task[0]

                    if resource_type == "stone":
                        if np.abs(obs[agent][4] - max_resource_inventory) < 1e-4:
                            tasks_done[agent] = task[0]

                    # If the building we are trying to build no longer requires the resource, we would return True
                    resources_required = None
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_for_resource+2] -
                              0.01) < 1e-4:
                        resources_required = house["cost"][resource_type]/100
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_for_resource+2] -
                              0.02) < 1e-4:
                        resources_required = mansion["cost"][resource_type]/100
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_for_resource+2] -
                              0.03) < 1e-4:
                        resources_required = castle["cost"][resource_type]/100

                    # The resources used to build the building so far
                    resources_used = None
                    resources_collected = None
                    if resource_type == "wood":
                        resources_used = obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT +
                                                    6*building_for_resource+3]
                        resources_collected = obs[agent][3]

                    if resource_type == "stone":
                        resources_used = obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT +
                                                    6*building_for_resource+4]
                        resources_collected = obs[agent][4]

                    # We would need to also consider the other agents performing the same task
                    for other_agent in self.agents:
                        if other_agent != agent:
                            other_agent_task = tasks[other_agent]
                            if other_agent_task and other_agent_task == task:
                                if resource_type == "wood":
                                    resources_collected += obs[other_agent][3]
                                if resource_type == "stone":
                                    resources_collected += obs[other_agent][4]

                    # If the sum of the resources used for construction and resources already collected by the
                    # agent is greater than or equal to the resources required to build the building, we would
                    # mark the collection task as done
                    if resources_used + resources_collected - resources_required >= 1e-4:
                        tasks_done[agent] = task[0]

                    # We don't really consider the resource blocks when considering the task done criteria
                    # for the collection task since the abstract state observation would always output the
                    # closest resource block to the agent which has the resource type the agent needs to collect
                    # available


                if task[0] == "build":
                    # We would check if the building has been built
                    building_id = task[1] - 1
                    built_status = obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+5]
                    if np.abs(built_status - 0) > 1e-4:
                        tasks_done[agent] = task[0]

            else:
                tasks_done[agent] = None

        return tasks_done



    def is_plan_valid(self, obs, tasks):
        """
        The method to check whether the plan is valid. We need to verify if the current tasks are possible to achieve
        by the agents as they currently exist
        :param obs: The current observation
        :param tasks: The tasks for the agents
        :return: True if the plan is valid, False otherwise
        """
        disrupted_agents = []
        for agent, task in tasks.items():
            if task:
                if task[0] == "collect":
                    resource_type = task[1]
                    building_id = task[2] - 1

                    # We would look at what kind of building we need to build and the resources required to build it
                    resources_required = None
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+2] - 0.01) < 1e-4:
                        resources_required = house["cost"]
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+2] - 0.02) < 1e-4:
                        resources_required = mansion["cost"]
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+2] - 0.03) < 1e-4:
                        resources_required = castle["cost"]

                    resources_required = {resource: resources_required[resource]/100
                                          for resource in resources_required.keys()}

                    # We would also look at the total resources used to build the building so far
                    resources_used = {}
                    resources_used["wood"] = obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+3]
                    resources_used["stone"] = obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+4]

                    # We would count the total resources of the resource type collected by the different agents
                    # performing the same task
                    resources_collected = {}
                    for agent in self.agents:
                        if tasks[agent] and tasks[agent] == task:
                            resources_collected["wood"] = obs[agent][3]
                            resources_collected["stone"] = obs[agent][4]

                    # If we need to collect the resource type for the building, we would need to check the different
                    # resources which are available in the environment
                    resources_collected_or_used_for_building = (resources_used[resource_type] +
                                                                resources_collected[resource_type])

                    # If we need to collect more resources for the building, we would need to check the resources
                    # available in the environment
                    if resources_required[resource_type] - resources_collected_or_used_for_building > 1e-4:
                        resource_available = 0
                        for i in range(RESOURCE_TILES_COUNT):
                            if np.abs(obs[agent][5*len(self.agents)+4*i+2] - 0.01) < 1e-4:
                                resource_available += obs[agent][5*len(self.agents)+4*i+3]
                            if np.abs(obs[agent][5*len(self.agents)+4*i+2] - 0.02) < 1e-4:
                                resource_available += obs[agent][5*len(self.agents)+4*i+3]

                        if resource_available < resources_required[resource_type] - \
                                                        resources_collected_or_used_for_building:
                            disrupted_agents.append(agent)


                if task[0] == "build":
                    # We would check if the current agent has any resources available to build the building
                    building_id = task[1] - 1

                    # We would look at what kind of building we need to build and the resources required to build it
                    resources_required = None
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+2] - 0.01) < 1e-4:
                        resources_required = house["cost"]
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+2] - 0.02) < 1e-4:
                        resources_required = mansion["cost"]
                    if np.abs(obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+2] - 0.03) < 1e-4:
                        resources_required = castle["cost"]

                    resources_required = {resource: resources_required[resource]/100
                                          for resource in resources_required.keys()}

                    # We would also look at the total resources used to build the building so far
                    resources_used = {}
                    resources_used["wood"] = obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+3]
                    resources_used["stone"] = obs[agent][5*len(self.agents)+4*RESOURCE_TILES_COUNT + 6*building_id+4]

                    # We would return False if the agent does not have a single resource of a type required to build
                    # the building
                    any_build_possible = False
                    for resource in resources_required.keys():
                        resource_need_to_collect = resources_required[resource] - resources_used[resource]
                        if resource_need_to_collect  > 1e-4:
                            if obs[agent][3] > 1e-4 and resource == "wood":
                                any_build_possible = True
                            if obs[agent][4] > 1e-4 and resource == "stone":
                                any_build_possible = True

                    # If no build action is possible, we would return False
                    if not any_build_possible:
                        disrupted_agents.append(agent)


        return len(disrupted_agents) == 0, disrupted_agents



    def get_abstract_obs(self, obs, tasks):
        # We would get the abstract observation for the different agents
        # Collect Task
        #   (x, y) for the agent
        #   Orientation
        #   Resource inventory count
        #   x, y for the other agents
        #   (x, y for the resource tile)
        #   Count for the resource tile

        # Build Task
        #   (x, y) for the agent
        #   Orientation
        #   Resource inventory count w, s
        #   x, y for the other agents
        #   x, y for the building

        abstract_obs_dict = {agent: np.zeros((5 + 2*len(self.agents), )) for agent in self.agents}
        for agent, task in tasks.items():
            if task:
                if task[0] == "collect":
                    # We would get the agent's position (x, y), orientation for the agent
                    abstract_obs_dict[agent][0] = obs[agent][0]
                    abstract_obs_dict[agent][1] = obs[agent][1]
                    abstract_obs_dict[agent][2] = obs[agent][2]

                    # The next value would be the type of resource collected by the agent
                    if task[1] == "wood":
                        abstract_obs_dict[agent][3] = obs[agent][3]
                    if task[1] == "stone":
                        abstract_obs_dict[agent][3] = obs[agent][4]

                    # We would first get the x, y locations for the other agents
                    other_agents = [a for a in self.agents if a != agent]
                    for i, other_agent in enumerate(other_agents):
                        abstract_obs_dict[agent][4+2*i] = obs[other_agent][0]
                        abstract_obs_dict[agent][4+2*i+1] = obs[other_agent][1]

                    # Once, we have the other agents' positions, we would get the x, y for the resource tile of the
                    # type closest to the agent, and of the same type as the resource to be collected
                    closest_resource_tile_available = None
                    closest_resource_tile_distance = np.inf
                    for resource_tile in range(RESOURCE_TILES_COUNT):
                        # If the count of the resource tile is not 0, we would get the distance of the resource tile
                        # from the agent
                        if obs[agent][5*len(self.agents)+resource_tile*4+3] > 1e-4:
                            if      (np.abs(obs[agent][5*len(self.agents)+resource_tile*4+2] -0.01) < 1e-4 \
                                    and task[1] == "wood") or \
                                    (np.abs(obs[agent][5*len(self.agents)+resource_tile*4+2] -0.02) < 1e-4 \
                                    and task[1] == "stone"):

                                x_dist = obs[agent][0] - obs[agent][5*len(self.agents) + resource_tile*4]
                                y_dist = obs[agent][1] - obs[agent][5*len(self.agents)+resource_tile*4+1]
                                distance = np.sqrt(x_dist**2 + y_dist**2)

                                if distance < closest_resource_tile_distance:
                                    closest_resource_tile_distance = distance
                                    closest_resource_tile_available = resource_tile

                    if closest_resource_tile_available is not None:
                        # We would get the x, y for the resource tile
                        abstract_obs_dict[agent][4 + 2*len(other_agents)] = obs[agent][5*len(self.agents)
                                                                                +closest_resource_tile_available*4]
                        abstract_obs_dict[agent][4 + 2*len(other_agents)+1] = obs[agent][5+4*len(self.agents)
                                                                                +closest_resource_tile_available*4+1]
                        # We would get the count for the resource tile
                        abstract_obs_dict[agent][4 + 2*len(other_agents)+2] = obs[agent][5+4*len(self.agents)
                                                                                +closest_resource_tile_available*4+3]

                if task[0] == "build":
                    # We would get the agent's position (x, y), orientation for the agent
                    abstract_obs_dict[agent][0] = obs[agent][0]
                    abstract_obs_dict[agent][1] = obs[agent][1]
                    abstract_obs_dict[agent][2] = obs[agent][2]

                    # The next value would be the count of type of resource collected by the agent
                    abstract_obs_dict[agent][3] = obs[agent][3]
                    abstract_obs_dict[agent][4] = obs[agent][4]

                    # We would first get the x, y locations for the other agents
                    other_agents = [a for a in self.agents if a != agent]
                    for i, other_agent in enumerate(other_agents):
                        abstract_obs_dict[agent][5+2*i] = obs[other_agent][0]
                        abstract_obs_dict[agent][5+2*i+1] = obs[other_agent][1]

                    # Once, we have the other agents' positions, we would get the x, y for the building
                    building_id = task[1]
                    abstract_obs_dict[agent][4 + 2*len(other_agents)] = obs[agent][5*len(self.agents)
                                                                            +4*RESOURCE_TILES_COUNT+6*(building_id-1)]
                    abstract_obs_dict[agent][4 + 2*len(other_agents)+1] = obs[agent][5*len(self.agents)
                                                                            +4*RESOURCE_TILES_COUNT+6*(building_id-1)+1]

            else:
                # We set the first 5 values to the agent's position, orientation and resource inventory count
                abstract_obs_dict[agent][:5] = obs[agent][:5]

        return abstract_obs_dict


    def get_observation_space_dict(self, obs_dict):
        """
        The method to get the observation space dict for the different agents
        :param obs_dict: The observation dict for the different agents
        :return: The observation space dict for the different agents
        """
        observation_size = 5 + 2*len(self.agents)
        for agent in self.agents:
            obs_dict[agent] = Box(low= 0, high=1, shape = (observation_size, ), dtype= np.float32)
        return obs_dict

    def get_action_space_dict(self, action_dict):
        """
        The method to get the action space dict for the different agents
        :param action_dict: The action space dict for the different agents
        :return: The action space dict for the different agents
        """
        return action_dict

    def get_plan(self, obs_dict):
        """
        The method to get the plan for the different agents
        :param obs_dict: The observation dict for the different agents
        :return: The plan for the different agents
        """

        # We need to get the current state and the goal from the environment
        state, goal = get_environment_state(obs_dict)

        # If there are no more goals remaining, we would return None
        if len(goal.to_build) == 0:
            return None

        # We would compute the single agent plan using PyHop
        plan = hop.pyhop(state, [("achieve_goal", goal)])

        # If no plan is found, we would exit
        if not plan:
            print("State")
            plan = hop.pyhop(state, [("achieve_goal", goal)], verbose=3)
            print(state.agents)
            print(state.resources)
            print(state.buildings)
            print("\n\nGoal")
            print(goal.to_build)
            print(obs_dict)
            sys.exit("No plan found. Try Again!")

        # We need to copy the state and execute the plan as we need to know which agents are involved in the
        # construction of the building
        temp_state = copy.deepcopy(state)
        for task in plan:
            if task and task[0] == "collect":
                temp_state = collect(temp_state, task[1], task[2])
            if task and task[0] == "build":
                temp_state = build(temp_state, task[1])

        # We will get the agents involved in the construction of each building
        agents_involved_in_construction = {building: [] for building in temp_state.buildings.keys()}
        for agent in temp_state.agents.keys():
            for building in temp_state.buildings.keys():
                if temp_state.buildings[building]["agents_involved_in_construction"][agent]:
                    agents_involved_in_construction[building].append(agent)

        # The tasks are aggregated based on the buildings the tasks are for.
        # One additional change is that we would introduce a build task for each additional collect
        # resource task of a type for the same building
        aggregated_tasks = aggregate_tasks(plan, collect_and_build_task_aggregator)

        # We would compute the duration of the tasks
        task_duration = calculate_task_duration(aggregated_tasks)

        # We would compute the distributed plans for the different agents
        distributed_plan = create_distributed_plans(task_duration_dict=task_duration, num_agents=len(self.agents))

        # We would now need to assign the different plans to the different agents based on the distributed plan
        # The only thing we need to check is the agent inventory. We can't assign an agent a plan, if it has a collect
        # task for a resource but the agent inventory for the resource is full
        # We would check which agents have such restrictions
        agent_resource_inventory_full = {agent: {resource: False for resource in ["wood", "stone"]}
                                         for agent in self.agents}
        max_resource_inventory = np.round(MAX_INVENTORY_PER_RESOURCE/100, 2)
        for agent in self.agents:
            for resource in ["wood", "stone"]:
                if np.abs(state.agents[agent]["resources"][resource] - max_resource_inventory) < 1e-4:
                    agent_resource_inventory_full[agent][resource] = True

        # We would iterate over the different plans and check what resources are collected before the first build task
        plan_resources_to_collect_before_first_build = {p_id: {resource: 0 for resource in ["wood", "stone"]}
                                                        for p_id in distributed_plan.keys()}

        # We may need to do additional processing where if
        for plan_id, plan in distributed_plan.items():
            for task in plan:
                if task[0] == "collect":
                    resource = task[1]
                    plan_resources_to_collect_before_first_build[plan_id][resource] += 1

                if task[0] == "build":
                    break

        # We would now assign the plans to the different agent
        agent_plans = {agent: [] for agent in self.agents}
        plans_assigned = {plan_id: False for plan_id in distributed_plan.keys()}
        for agent in agent_plans.keys():
            for plan_id, plan in distributed_plan.items():
                if not plans_assigned[plan_id]:
                    can_assign_plan = True
                    # We would check if the agent has a full inventory for any resource
                    for resource in ["wood", "stone"]:
                        if agent_resource_inventory_full[agent][resource] and \
                                plan_resources_to_collect_before_first_build[plan_id][resource] > 1e-4:
                            can_assign_plan = False
                            continue

                    if can_assign_plan:
                        agent_plans[agent] = plan
                        plans_assigned[plan_id] = True
                        break

        # We would try to assign the remaining plans to the agents which have not been assigned a plan
        for plan_id, plan in distributed_plan.items():
            if not plans_assigned[plan_id]:
                for agent in self.agents:
                    if not plans_assigned[plan_id]:
                        agent_plans[agent] = plan
                        plans_assigned[plan_id] = True
                        break
        # We need to make sure that all the agents which were involved in a particular building construction were
        # assigned the build task for the building
        for building, agents in agents_involved_in_construction.items():
            for agent in agents:
                b_id = int(building.split("_")[1])
                if not agent_plans[agent]:
                    agent_plans[agent] = [("build", b_id)]

                if ("build", b_id) not in agent_plans[agent]:
                    # In this case, we would need to insert the build task at the start of the plan to ensure that
                    # we build the building as soon as possible
                    agent_plans[agent].insert(0, ("build", b_id))

        # We want to perform a sort of the plans by the b_id of the building to ensure that the agents are building
        # the buildings in the correct order
        for agent in agent_plans.keys():
            agent_plans[agent] = sorted(agent_plans[agent], key=lambda x: x[-1])

        self.plan = agent_plans

        return self.plan

    def get_operators(self):
        """
        The method to get the operator for the planner
        :return: The operator for the planner
        """
        return self.operators