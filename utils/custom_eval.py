import ray
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

import pdb

def multi_mdp_eval(algorithm, eval_workers):

    # We configured 2 eval workers in the training config.
    worker_1, worker_2, worker_3, worker_4, worker_5  = eval_workers.remote_workers()
    # DEFAULT_CONFIG["eval"]=True
    worker_1.foreach_env.remote(lambda env: env.set_seed(38))
    worker_2.foreach_env.remote(lambda env: env.set_seed(39))
    worker_3.foreach_env.remote(lambda env: env.set_seed(40))
    worker_4.foreach_env.remote(lambda env: env.set_seed(41))
    worker_5.foreach_env.remote(lambda env: env.set_seed(43))
    for i in range(2):
        print("Custom evaluation round", i+1)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    metrics = {}
    

    episodes = collect_episodes(workers=eval_workers)
    pdb.set_trace()
    result = summarize_episodes(episodes)
    
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

    metrics=result

    return metrics
