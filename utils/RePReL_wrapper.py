from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict


class RePReLWrapper(MultiAgentEnv):

    def __init__(self,
                 env,
                 planner,
                 use_heuristic = False,
                 terminal_reward=10):
        super().__init__()
        self.env = env
        self.planner = planner
        self.agent_names = env.get_agent_ids
        self._policy_ids = None
        self.user_data_fields = env.user_data_fields
        self.heuristic_based = use_heuristic
        self.heuristic = self.env.calculate_heuristic if use_heuristic else None

        if planner:
            planner_agents = planner.get_agents()
            planner_tasks = planner.get_operators()
            # The policy ids are of the form agent_task, we would have agent*task policy ids
            self._policy_ids = [f"{agent}_{task}" for agent in planner_agents for task in planner_tasks]

        self.terminal_reward = terminal_reward
        self._current_tasks = None
        self.agent_task_index = dict()
        obs = self.reset()

        # Mapping from the agent observation space to the policy observation space
        self.observation_space_dict = dict()
        agent_obs_dict = self.planner.get_observation_space_dict(env.observation_space_dict)
        for agent in planner_agents:
            for task in planner_tasks:
                self.observation_space_dict[f"{agent}_{task}"] = agent_obs_dict[agent]

        # There is no need for any mapping as the get action space dict returns the action space dict for the policy ids
        agent_action_dict = self.planner.get_action_space_dict(env.action_space_dict)
        self.action_space_dict = dict()
        for agent in planner_agents:
            self.action_space_dict[agent] = 0
            for task in planner_tasks:
                self.action_space_dict[f"{agent}_{task}"] = agent_action_dict[agent]

    def get_agent_ids(self):
        return self._policy_ids

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = self._policy_ids

        # We need to map the agent_ids to actual agents before we are able to sample from the environment
        agent_ids_for_base = list(set([agent_id.split("_")[0] for agent_id in agent_ids]))
        env_actions =  self.env.action_space_sample(agent_ids_for_base)
        policy_actions = dict()
        for agent_id in agent_ids:
            agent = agent_id.split("_")[0]
            policy_actions[agent_id] = env_actions[agent]

        return policy_actions

    def reset(self):
        # We would reset the environment and the planner, and return the abstract observation dictionaries based on
        # those new observations
        obs = self.env.reset()
        self.planner.reset(obs)
        self.agent_task_index = dict()

        # We would also need to reset the current task indices for the agents to 0
        for agent in self.planner.get_agents():
            self.agent_task_index[agent] = 0

        self._current_tasks = self.planner.next_tasks(obs)
        abstract_obs_dict = dict()
        for agent, abstract_obs in self.planner.get_abstract_obs(obs, self._current_tasks).items():
            task = self._current_tasks[agent]
            if task:
                abstract_obs_dict[f"{agent}_{task[0]}_{self.agent_task_index[agent]}"] = abstract_obs

        return abstract_obs_dict


    def step(self, action):

        # We need to check whether the actions sent to the gym environment correspond to action or policy_ids
        temp = dict()
        for policy_or_agent_id in action.keys():
            split = policy_or_agent_id.split("_")
            temp[split[0]] = action[policy_or_agent_id]

        action = temp
        obs, reward, done, info = self.env.step(action)
        agent_task_done = self.planner.is_task_done(obs, self._current_tasks)
        plan_valid, disrupted_agents = self.planner.is_plan_valid(obs, self._current_tasks)

        # if not plan_valid:
        #     print("Plan is not valid. Will have to change it")
        #     print("Current plan: ", self.planner.get_current_plan())

        # Initialize the task_done dictionary to False for each policy agent
        task_done = {f"{policy_id}_{self.agent_task_index[policy_id.split('_')[0]]}"
                     : False for policy_id in self._policy_ids}

        # The new_tasks flag would determine whether we would need to call the next tasks method of the planner
        new_tasks = False

        for agent, task in agent_task_done.items():
            # Since, task would return the task string in case of a task completion, and False otherwise
            if task:
                new_tasks = True
                task_done[f"{agent}_{task}_{self.agent_task_index[agent]}"] = True

        # We update the rewards for the agents based on whether the task is done or not
        policy_rewards = dict()
        for policy_agent_indexed, is_done in task_done.items():
            # If the task is done, we give the terminal reward
            if policy_agent_indexed not in policy_rewards.keys():
                policy_rewards[policy_agent_indexed] = 0

            if is_done:
                a, t, i = policy_agent_indexed.split("_")
                policy_rewards[policy_agent_indexed] = self.terminal_reward + reward[a]
            else:
                # If the task is not done, but the policy_agent is used, we use the same reward as the agent
                # Otherwise, we do not update the reward
                a, t, i = policy_agent_indexed.split("_")
                if t is not None and a in self._current_tasks.keys() and self._current_tasks[a] is not None \
                        and t in self._current_tasks[a][0]:
                    policy_rewards[policy_agent_indexed] += reward[a]

                    #Heuristic Added here
                    if not plan_valid and self.heuristic_based and a in disrupted_agents:
                        heuristic = self.env.calculate_heuristic(a,self._current_tasks[a][1])
                        intrinsic_reward = self.terminal_reward/(heuristic+1) if heuristic<10 else 0
                        policy_rewards[policy_agent_indexed] += intrinsic_reward

                else:
                    # We remove any rewards that are not used in the current step
                    del policy_rewards[policy_agent_indexed]

        # We get whether all the tasks are done from the environment step
        task_done["__all__"] = done["__all__"]

        # We map the abstract_obs from the agent abstract observation space to the policy abstract observation space
        abstract_obs_dict = self.planner.get_abstract_obs(obs, self._current_tasks)
        policy_obs = dict()
        policy_info = dict()

        for agent, abstract_obs in abstract_obs_dict.items():
            task = self._current_tasks[agent]
            if task:
                policy_obs[f"{agent}_{task[0]}_{self.agent_task_index[agent]}"] = abstract_obs

                # We also update the info to have the same keys as the abstract_obs
                policy_info[f"{agent}_{task[0]}_{self.agent_task_index[agent]}"] = info[agent]

        # We need to remove the extraneous keys from the done and the rewards dictionary
        task_done_keys = list(task_done.keys())
        for key in task_done_keys:
            if key not in policy_obs.keys() and key != "__all__":
                del task_done[key]

        policy_rewards_keys = list(policy_rewards.keys())
        for key in policy_rewards_keys:
            if key not in policy_obs.keys():
                del policy_rewards[key]

        # If there are some tasks that have been finished, we need to compute the new tasks
        if (new_tasks or not plan_valid) and not done["__all__"]:
        
            
            old_tasks = self._current_tasks
            self._current_tasks = self.planner.next_tasks(obs)


            # print("New Plan: ", self.planner.get_current_plan())

            # We also get the abstract observations for the new tasks
            abstract_obs_dict = self.planner.get_abstract_obs(obs, self._current_tasks)

            # For agents for which the current tasks have changed, being it finished or not, we need to add those
            # entries to our policy_obs, policy_rewards, task_done and info
            for agent, task in self._current_tasks.items():
                if task and task != old_tasks[agent]:
                    # Update the task done for the old task to true
                    if old_tasks[agent]:
                        task_done[f"{agent}_{old_tasks[agent][0]}_{self.agent_task_index[agent]}"] = True

                    # Update the agent_task_index if the next task is different from the current task
                    self.agent_task_index[agent] += 1
                    # Update the policy_obs
                    policy_obs[f"{agent}_{task[0]}_{self.agent_task_index[agent]}"] = abstract_obs_dict[agent]
                    policy_rewards[f"{agent}_{task[0]}_{self.agent_task_index[agent]}"] = 0
                    task_done[f"{agent}_{task[0]}_{self.agent_task_index[agent]}"] = False
                    if old_tasks[agent]:
                        policy_info[f"{agent}_{task[0]}_{self.agent_task_index[agent]}"] \
                            = policy_info[f"{agent}_{old_tasks[agent][0]}_{self.agent_task_index[agent] - 1}"]
                    else:
                        policy_info[f"{agent}_{task[0]}_{self.agent_task_index[agent]}"] = info[agent]

                if task is None and old_tasks[agent]:
                    task_done[f"{agent}_{old_tasks[agent][0]}_{self.agent_task_index[agent]}"] = True

        # If the task_done["__all__"] is True, we need to set the task_done for all the tasks to True
        task_done["__all__"] = done["__all__"]
        if task_done["__all__"]:
            for key in task_done.keys():
                if key != "__all__":
                    task_done[key] = True

        return policy_obs, policy_rewards, task_done, policy_info

    def render(self, **kwargs):
        return self.env.render(**kwargs)