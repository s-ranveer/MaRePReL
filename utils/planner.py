from abc import ABC, abstractmethod


class Planner(ABC):

    @abstractmethod


    def __init__(self, obs):
        """Initialize the Planner"""

    @abstractmethod
    def reset(self, obs):
        """Reset the planner"""

    @abstractmethod
    def next_tasks(self, obs):
        """Get next task for each agent"""

    @abstractmethod
    def is_task_done(self, obs, tasks):
        """Check if the task is done"""

    @abstractmethod
    def get_abstract_obs(self, obs, tasks):
        """Abstract observation for each task and agent"""

    @abstractmethod
    def get_observation_space_dict(self, obs_dict):
        """Define observation space dict for each agent"""

    @abstractmethod
    def get_action_space_dict(self, obs_dict):
        """Define action space dict for each agent"""

