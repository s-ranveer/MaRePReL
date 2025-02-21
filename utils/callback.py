from typing import Dict, Optional, Union
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.tune.logger import LoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.utils.util import SafeFallbackEncoder
import os
import numpy as np 
from functools import reduce

from copy import deepcopy
import os
import os.path as osp
from collections import namedtuple
from pathlib import Path
import git

GitInfo = namedtuple(
    'GitInfo',
    [
        'directory',
        'code_diff',
        'code_diff_staged',
        'commit_hash',
        'branch_name',
    ],
)


def project_root():
    return os.environ.get("PROJECT_ROOT_DIR", os.getcwd())


project_dir = Path(project_root())


def get_git_infos(dirs):
    git_infos = None
    try:
        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError as e:
                print("Not a valid git repo: {}".format(directory))
    except ImportError:
        git_infos = None
    return git_infos



def log_git(log_dir, code_dirs=None):
    if code_dirs is None:
        code_dirs = [project_dir]
    git_infos = get_git_infos(code_dirs)
    if git_infos is not None:
        for (
                directory, code_diff, code_diff_staged, commit_hash, branch_name
        ) in git_infos:
            directory = str(directory)
            if directory[-1] == '/':
                directory = directory[:-1]
            diff_file_name = directory[1:].replace("/", "-") + ".patch"
            diff_staged_file_name = (
                    directory[1:].replace("/", "-") + "_staged.patch"
            )
            if code_diff is not None and len(code_diff) > 0:
                with open(osp.join(log_dir, diff_file_name), "w") as f:
                    f.write(code_diff + '\n')
                # logger.info(code_diff)
            if code_diff_staged is not None and len(code_diff_staged) > 0:
                with open(osp.join(log_dir, diff_staged_file_name), "w") as f:
                    f.write(code_diff_staged + '\n')
                # logger.info(code_diff_staged)
            with open(osp.join(log_dir, "git_infos.txt"), "a") as f:
                f.write("directory: {}\n".format(directory))
                f.write("git hash: {}\n".format(commit_hash))
                f.write("git branch name: {}\n\n".format(branch_name))
            # logger.info("directory: {}\n".format(directory))
            # logger.info("git hash: {}\n".format(commit_hash))
            # logger.info("git branch name: {}\n\n".format(branch_name))



class MultiagentCallbacks(DefaultCallbacks):

    def __init__(self, transfer=True):
        self.ma_episode_diagnostics ={} 
        self.transfer=  transfer
        self.transfer_init_step = 3000000
        
        
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        
        print(algorithm)
        log_git(algorithm.logdir)
        print("Logger dir: {}".format(algorithm.logdir))



    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        # Initialize a unique random seed for the worker
        worker_index = worker.worker_index
        seed_base = np.random.randint(0,100000) # Base seed (adjust as needed)
        seed = seed_base + worker_index
        self.env = base_env.get_sub_environments()[0]
        try:
            
            self.env.set_seed(seed)
        except:
            self.env.env.set_seed(seed)
        episode.custom_metrics["Seed"] = seed  # Optional: Store seed as a custom metric
     


    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        
        try:
            metrics = base_env.get_sub_environments()[0].target_diagnostics
            episode_diagnostics = base_env.get_sub_environments()[0].get_diagnostics()
        except:
            metrics = base_env.get_sub_environments()[0].env.target_diagnostics
            episode_diagnostics = base_env.get_sub_environments()[0].env.get_diagnostics()
        # print(metrics)
        
        # print(episode_diagnostics)
        for i in metrics:

            episode.custom_metrics[i] = episode_diagnostics[i]
            episode.hist_data[i] = [episode.custom_metrics[i]]
        episode.custom_metrics[i]



    def on_train_result(self, *, result, **kwargs):

        #OfficeWorldSpecific
        if "Visit_Tasks" in result["hist_stats"]:
                result["Visit%"] = sum(result["hist_stats"]["Visits_Completed"])/sum(result["hist_stats"]["Visit_Tasks"])
        
        if "Pickup_Tasks" in result["hist_stats"]:
                result["Pickup%"] = sum(result["hist_stats"]["Pickups_Completed"])/sum(result["hist_stats"]["Pickup_Tasks"])
        
        if "Deliver_Tasks" in result["hist_stats"]:
                result["Deliver%"] = sum(result["hist_stats"]["Delivers_Completed"])/sum(result["hist_stats"]["Deliver_Tasks"])


        #TaxiSpecific
        if "passengers_picked" in result["hist_stats"] and 'total_tasks' in result["hist_stats"]:
            result["Pickup%"] = sum(result['hist_stats']['passengers_picked'])/ sum(result["hist_stats"]["total_tasks"])
        
        if "passengers_dropped" in result["hist_stats"] and 'total_tasks' in result["hist_stats"]:
            result["Drop%"] = sum(result['hist_stats']['passengers_dropped'])/ sum(result["hist_stats"]["total_tasks"])
        

        #DungeonSpecific
        if "Number_of_Agent_Deaths" in result['hist_stats']:
            result["Death%"]= (sum(result['hist_stats']['Number_of_Agent_Deaths'])/
                               sum(result['hist_stats']['Total_Agents']))

        if "Number_of_Enemy_Kills" in result['hist_stats']:
            result["Kill%"]= (sum(result['hist_stats']['Number_of_Enemy_Kills'])/
                               sum(result['hist_stats']['Total_Enemies']))

        if "Number_of_Keys_in_Door" in result['hist_stats']:
            result["Unlock%"]= (sum(result['hist_stats']['Number_of_Keys_in_Door'])/
                               sum(result['hist_stats']['Total_Keys']))
            
        if self.transfer:
            result["timesteps_since_transfer"] = result["timesteps_total"] - self.transfer_init_step

