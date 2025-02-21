# The aggregator to be used for the dungeon domain
import copy
def dungeon_task_aggregator(tasks, state):
    """
    This function aggregates the tasks for the dungeon domain. The aggregation is done based on the
    objects held by the agents. The agents holding the same object would be aggregated together.
    :param state: The current state of the environment
    :param tasks: The list of tasks to be executed
    """
    # We would aggregate the get_key tasks with the same agents holding the key together
    agent_keys = {}
    for key, value in state.enemies.items():
        if value["key"] != 1 and value["key"] != 0:
            if value["key"] not in agent_keys.keys():
                agent_keys[value["key"]] = [key]
            else:
                agent_keys[value["key"]].append(key)

    # Once, we have the list of the objects held by each agent, we would aggregate their add_key tasks
    # together
    aggregated_tasks = {}
    temp = copy.deepcopy(tasks)
    for key, value in agent_keys.items():
        for i, task in enumerate(tasks):
            # If the task is get_key_in_door
            if task[0] == "get_key_in_door":
                # The key is held by the agent
                if task[1] in value:
                    # We would add the task to the aggregated task and remove it from the temporary task
                    # list
                    if key not in aggregated_tasks.keys():
                        aggregated_tasks[key] = [task]
                    else:
                        aggregated_tasks[key].append(task)
                    temp.pop(i)

    # We would add the remaining tasks to the aggregated task list and return them
    return list(aggregated_tasks.values()) + temp




