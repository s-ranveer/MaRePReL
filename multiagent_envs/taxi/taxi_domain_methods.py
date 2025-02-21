import numpy as np
from task_scheduling.nodes import ScheduleNode


# The aggregator for the taxi domain
def taxi_task_aggregator(tasks):
    # For the taxi domain, the pickup and drop for the same agent can be aggregated together

    # Get all the unique passengers ids
    unique_passengers = set([task[1] for task in tasks])

    # For each unique passenger id, we aggregate the pickup and drop tasks
    aggregated_tasks = []
    for agent in unique_passengers:
        # Get all the pickup and drop tasks for the passenger
        passenger_tasks = [task for task in tasks if task[1] == agent]
        # Aggregate the tasks
        aggregated_tasks.append(passenger_tasks)

    return aggregated_tasks

# The method would calculate the length of each task for the taxi domain
# This heuristic is not correct, it needs to be completely rewritten for it to make any sense
# TODO: Rewrite the heuristic for the taxi domain

# def taxi_task_duration(task, taxi_locations=None, pickup_locations=None, drop_locations=None):
#     """
#     This method would calculate the duration of the task based on the observation
#     """
#     if (taxi_locations is None) or (pickup_locations is None) or (drop_locations is None):
#         raise ValueError("The taxi locations, pickup locations and drop locations cannot be None")
#
#     total_duration = 0
#     # Iterate over each subtask of the task to calculate the duration
#     for subtask in task:
#         action, passenger_id = subtask
#
#         # Get the pickup and drop locations for the passenger
#         passenger_pickup_location = pickup_locations[passenger_id - 1]
#         passenger_drop_location = drop_locations[passenger_id - 1]
#
#         taxi_distance = 0
#         if action == "pickup":
#             # Get the pickup location of the passenger
#             passenger_pickup_location = pickup_locations[passenger_id - 1]
#
#             # For each taxi, compute the distance between the passenger and the taxi
#             for taxi in taxi_locations:
#                 taxi_distance += np.linalg.norm(taxi - passenger_pickup_location, ord=1)
#
#             total_duration += taxi_distance / len(taxi_locations)
#
#         if action == "drop":
#             # For each taxi, compute the distance between the passenger and the taxi
#
#             for taxi in taxi_locations:
#                 taxi_distance += np.linalg.norm(passenger_drop_location - passenger_pickup_location, ord=1)
#
#             total_duration += taxi_distance / len(taxi_locations)
#
#     return total_duration
