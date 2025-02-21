def collect_and_build_task_aggregator(tasks):
    """
    This method aggregates the different tasks for the collect and build domain. The aggregation is done based on the
    building they are building
    """
    # For each building, we would calculate the number of times different resources are collected
    building_resources = {}
    collect_tasks_found = False
    for task in tasks:
        if task[0] == "collect":
            collect_tasks_found = True
            resource_type = task[1]
            building = task[2]
            if building not in building_resources.keys():
                building_resources[building] = {resource_type: 1}
            else:
                if resource_type not in building_resources[building].keys():
                    building_resources[building][resource_type] = 1
                else:
                    building_resources[building][resource_type] += 1

    # The way the planner works, build tasks for each type of building are always going to be present
    if collect_tasks_found:
        aggregated_tasks = []
        for building in building_resources.keys():
            while any(building_resources[building].values()):
                task = []
                for resource_type, count in building_resources[building].items():
                    if count > 0:
                        task.append(("collect", resource_type, building))
                        building_resources[building][resource_type] -= 1

                # Append the build task to the task list
                task.append(("build", building))
                aggregated_tasks.append(task)

    # If no collect tasks are found, we don't need to aggregate the tasks as the build tasks
    # are already present in the task list and can't be aggregated with other build tasks
    else:
        aggregated_tasks = []
        for task in tasks:
            aggregated_tasks.append([task])

    # If the aggregated tasks have a length of 1 and more than one collect task, we would split the collect tasks
    # into separate tasks

    # Can remove this if the planner causes issues with the current implementation
    if len(aggregated_tasks) == 1 and len(aggregated_tasks[0]) > 2:
        new_aggregated_tasks = []
        for task in aggregated_tasks[0]:
            if task[0] == "collect":
                new_aggregated_tasks.append([task, ("build", task[2])])
        aggregated_tasks = new_aggregated_tasks

    return aggregated_tasks
