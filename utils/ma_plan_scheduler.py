import numpy as np

from task_scheduling import algorithms
from task_scheduling.tasks import Generic

seed = 12345


# We first need to aggregate the tasks based on the agents that can do them
def aggregate_tasks(plan, aggregator=None, **kwargs):
    """
    This method would aggregate the tasks based on the aggregator provided
    :param plan: The plan provided
    :param aggregator: The current aggregator provided
    :param kwargs: The additional arguments to be provided to the aggregator
    :returns: The aggregated plan

    """
    # If the aggregator is not provided, we return the plan with each tasks
    if not aggregator:
        return plan

    # If the aggregator is provided, we aggregate the tasks based on the aggregator
    return aggregator(plan, **kwargs)


# We aggregate the tasks for the agent


# We need to calculate the duration for each task
def calculate_task_duration(tasks, heuristic=None, **kwargs):
    """
    This method would calculate the duration for each task

    :param tasks: The list of tasks
    :param heuristic: The heuristic to calculate the duration of the task
    :param kwargs: The additional arguments to be provided to the heuristic

    :returns: The dictionary of task durations

    """
    task_duration_dict = {}

    # If the heuristic is not provided, we assume that the duration for each task is the number of subtasks
    if not heuristic:
        task_duration_dict = {i: {"duration": len(tasks[i]), "task": tasks[i]} for i in range(len(tasks))}

    # When we are provided with a heuristic, we use it to calculate the duration of each task
    else:
        duration = [heuristic(tasks[i], **kwargs) for i in range(len(tasks))]
        task_duration_dict = {i: {"duration": duration[i], "task": tasks[i]} for i in range(len(tasks))}

    return task_duration_dict


# Once, we have the duration for each task, we can create the task objects
def create_distributed_plans(task_duration_dict, num_agents):
    """
    This method would create the distributed plans for the agents
    :param task_duration_dict: The dictionary of task durations
    :param num_agents: The number of agents

    :returns: The distributed plans for the agents

    """
    task_objects = []
    for key in task_duration_dict.keys():
        task_objects.append(
            Generic(duration=task_duration_dict[key]["duration"],
                    t_release=0,
                    name=key,
                    loss_func=lambda x: 0))

    # We need to set the channel availability (i.e. the number of agents)
    ch_avail = [0.5 * i for i in range(num_agents)]

    # Define and assess algorithms
    sch = algorithms.branch_bound(task_objects, ch_avail)

    # We sort the schedule based on the start time
    sch = sch.tolist()
    sch = [(k, i) for i, k in enumerate(sch)]
    sch.sort(key=lambda x: (x[0][1], x[0][0]))

    # We use the schedule to assign the tasks to each of the agents
    distributed_plans = dict()

    # We iterate over the schedule, assigning each task
    for i in range(len(sch)):
        _, agent_id = sch[i][0]
        if agent_id not in distributed_plans.keys():
            distributed_plans[agent_id] = []
        distributed_plans[agent_id].append(task_duration_dict[int(task_objects[sch[i][1]].name)]["task"])

    # We convert the plans to a flat list before returning them
    for key in distributed_plans.keys():
        distributed_plans[key] = [item for sublist in distributed_plans[key] for item in sublist]

    return distributed_plans



