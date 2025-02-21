# The task extended wrapper extends the basic environment wrapper
# to extend the observation for each agent to include its subplan emdedding using a transformer model
import numpy as np
from sentence_transformers import SentenceTransformer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import sys
EMBEDDING_DIM = 4
REDUCTION_TECHNIQUE = "mean"

class TaskExtendedWrapper(MultiAgentEnv):
    def __init__(self, env, planner):
        super().__init__()
        self.env = env
        self.planner = planner
        self.transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

        # We would create a buffer for storing the transformed plan text so that we do not call
        # the transformer over and over again

        if REDUCTION_TECHNIQUE == "mean":
            self.transformer_buffer = {"": np.zeros(EMBEDDING_DIM)}
        else:
            self.transformer_buffer = {}

        self.agent_names = env.get_agent_ids
        self.user_data_fields = env.user_data_fields
        self._current_agent_plans = {x:None for x in self.agent_names}

        #TODO: Modify the underlying functions to get the correct dict

        self.observation_space_dict = self.planner.get_observation_space_dict(env.observation_space_dict, mode="extended") 
        self.observation_space = self.observation_space_dict[self.agent_names[0]]
        self.action_space_dict = self.planner.get_action_space_dict(env.action_space_dict)
        self.action_space = self.action_space_dict[self.agent_names[0]]

    # The method for returning the current agent ids
    def get_agent_ids(self):
        return self.agent_names

    def action_space_sample(self, agent_ids: list = None):
        return self.env.action_space_sample(agent_ids)

    def render(self, **kwargs):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        # print("Observation size during reset ", len(obs))
        # print("Obs ", obs)
        self.planner.reset(obs)
        self._current_agent_plans = self.planner.next_tasks(obs, return_all_tasks=True)
        # print("Current Agent Plans ", self._current_agent_plans)
        extended_obs_dict = self.extend_observation_with_task_embeddings(obs, self._current_agent_plans)
        return extended_obs_dict
        

    def step(self, actions):
        # We would call the base environment with the agent actions
        observations, rewards, done, info = self.env.step(actions)
        # print("Step fn Obs ", observations)
        # We would check if the current task assigned to the agent has been done or the plan is valid
        # In case the task has been done or the plan is invalid, we would recompute the plan
    
        for agent in self.agent_names:
            if agent not in self._current_agent_plans:
                self._current_agent_plans[agent] = None
        

        current_tasks = {agent: self._current_agent_plans[agent][0] for agent in self.agent_names
                         if self._current_agent_plans[agent] is not None }



        missing_agents_from_tasks = set(self.agent_names) - set(current_tasks.keys())
        for agent in missing_agents_from_tasks:
            current_tasks[agent] = None

        tasks_done = self.planner.is_task_done(observations, current_tasks)
        plan_valid, disrupted_agents = self.planner.is_plan_valid(observations, current_tasks)

        # If there is any task done or the plan is not valid, we would recompute the plan
        if any(tasks_done.values()) or not plan_valid:
            self._current_agent_plans = self.planner.next_tasks(observations, return_all_tasks=True)

        # Once, we have the new plan, we would extend the observation with the new plan
        observations = self.extend_observation_with_task_embeddings(observations, self._current_agent_plans)

        return observations, rewards, done, info



    def extend_observation_with_task_embeddings(self, agent_obs_dict, planner_dict):
        for agent in agent_obs_dict.keys():
            if agent in planner_dict.keys():
                agent_plan = planner_dict[agent]

                # Once we have the agent plan, we would convert it to a string
                if agent_plan is None or agent_plan == []:
                    agent_plan  = ""
                else:
                    agent_plan = [f"{x[0]}_{x[1]}" for x in agent_plan]
                    agent_plan = " ".join(agent_plan)

                # Once we have the agent plan string, we would check if the embedding is already computed
                if agent_plan not in self.transformer_buffer.keys():
                    large_embedding = self.transformer_model.encode(agent_plan)
                    # We would reduce the dimensionality of the embedding to the required dimension by dividing
                    # the embedding into smaller chunks and then taking the mean of the chunks
                    if REDUCTION_TECHNIQUE == "mean":
                        reduced_embedding = np.zeros(EMBEDDING_DIM)
                        chunk_size = len(large_embedding) // EMBEDDING_DIM
                        for i in range(EMBEDDING_DIM):
                            reduced_embedding[i] = np.mean(large_embedding[i*chunk_size:(i+1)*chunk_size])
                    elif REDUCTION_TECHNIQUE == "truncated":
                        reduced_embedding = large_embedding[:EMBEDDING_DIM]

                    else:
                        reduced_embedding = large_embedding
                    # We would store the embedding in the buffer
                    self.transformer_buffer[agent_plan] = reduced_embedding

                # We would add the agent plan embedding to the observation
                agent_obs_dict[agent] = np.concatenate([agent_obs_dict[agent], self.transformer_buffer[agent_plan]])

            else:
                # TODO: Check if we need to add a zero vector or instead add the embedding of an empty plan
                agent_obs_dict[agent] = np.concatenate([agent_obs_dict[agent], self.transformer_buffer[""]])

        return agent_obs_dict


