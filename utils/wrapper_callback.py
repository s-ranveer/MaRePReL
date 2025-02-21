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

    def __init__(self):
        self.ma_episode_diagnostics ={} 

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
        env = base_env.get_sub_environments()[0].env
        env.set_seed(seed)
        episode.custom_metrics["Seed"] = seed  # Optional: Store seed as a custom metric
  
      


    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        

        episode_diagnostics = base_env.get_sub_environments()[0].env.get_diagnostics()
        episode.custom_metrics["Episode Length"] = int(episode_diagnostics["Episode Length"])
        episode.custom_metrics["Success"] = episode_diagnostics["Success"]
        episode.custom_metrics["Passengers Picked Up"] = int(episode_diagnostics["Passengers Picked Up"])
        episode.custom_metrics["Passengers Dropped"] = int(episode_diagnostics["Passengers Dropped"])
        episode.custom_metrics["Episode Return"] = float(episode_diagnostics["Episode Return"])
        episode.custom_metrics["Crash"] = episode_diagnostics["Crash"]
        episode.custom_metrics["Total Tasks"] = episode_diagnostics["Total Tasks"]

        episode.hist_data["Episode Length"] = [episode.custom_metrics["Episode Length"] ]
        episode.hist_data["Success"] = [episode.custom_metrics["Success"]]
        episode.hist_data["Passengers Picked Up"] = [episode.custom_metrics["Passengers Picked Up"]]
        episode.hist_data["Passengers Dropped"] = [episode.custom_metrics["Passengers Dropped"]]
        episode.hist_data["Episode Return"] = [episode.custom_metrics["Episode Return"]]
        episode.hist_data["Crash"] = [episode.custom_metrics["Crash"]]
        episode.hist_data["Total Tasks"] = [episode.custom_metrics["Total Tasks"]]


    def on_train_result(self, *, result, **kwargs):
        
        if not self.ma_episode_diagnostics:
            for agent in self.ma_episode_diagnostics.keys():
                for field, value in self.ma_episode_diagnostics[agent].items():
                    if(field.startswith("Passenger")):
                        result[agent+" "+field] = sum(result['hist_stats'][agent+ " " +field])/sum(result["hist_stats"]["Total Tasks"])
                    else: 
                        result[agent+" "+field] = sum(result['hist_stats'][agent+ " " +field])/(result['hist_stats'][agent+ " " +field])
            
        if "Success" in result['hist_stats']:
            result["Success%"]=sum(result['hist_stats']['Success'])/len(result['hist_stats']['Success'])

        if "Crash" in result['hist_stats']:
            result["Crash%"]= sum(result['hist_stats']['Crash'])/len(result['hist_stats']['Crash'])

        if "Passengers Picked Up" in result["hist_stats"] and "Total Tasks" in result["hist_stats"]:
            result["Pickup%"] = sum(result['hist_stats']['Passengers Picked Up'])/sum(result["hist_stats"]["Total Tasks"])

        if "Passengers Dropped" in result["hist_stats"] and "Total Tasks" in result["hist_stats"]:
            result["Drop%"] = sum(result['hist_stats']['Passengers Dropped'])/sum(result["hist_stats"]["Total Tasks"])
            
        if "Episode Return" in result["hist_stats"]:
            result["Avg. Return"] = sum(result['hist_stats']['Episode Return'])/len(result['hist_stats']['Episode Return'])
