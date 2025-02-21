
# The aggregator for the office world domain
def office_task_aggregator(tasks):

    # For the officeworld, the pickup and deliver can be aggegated together
    # if the object is the same
    unique_objects = set([task[1] for task in tasks])
    aggregated_tasks = []
    if ("visitOrPickup", 0) in tasks:
        if ("visitOrPickup", 1) in tasks:
            aggregated_tasks.append([("visitOrPickup", 0), ("visitOrPickup", 1)])
            tasks.remove(("visitOrPickup", 0))
            tasks.remove(("visitOrPickup", 1))
    if ("visitOrPickup", 2) in tasks:
        if ("visitOrPickup", 3) in tasks:
            aggregated_tasks.append([("visitOrPickup", 2), ("visitOrPickup", 3)])
            tasks.remove(("visitOrPickup", 2))
            tasks.remove(("visitOrPickup", 3))

    # Aggregate the tasks with the same objects
    for obj in unique_objects:
        aggregated_tasks.append([task for task in tasks if task[1] == obj])

    # We need to remove any lists which are empty
    aggregated_tasks = [task for task in aggregated_tasks if task != []]

    return aggregated_tasks