## MaRePReL
This repository contains an implementation of the paper <>

## Abstract
Multiagent Reinforcement Learning poses significant challenges due to the exponential growth of state, object, and action spaces and the non-stationary nature of multiagent environments. This results in notable sample inefficiency and hinders generalization across diverse tasks. The complexity is further pronounced in relational settings, where domain knowledge is crucial but often underutilized by existing MARL algorithms. To overcome these hurdles, we propose integrating relational planners as centralized controllers with efficient state abstractions and reinforcement learning. This approach proves to be sample-efficient and facilitates effective task transfer and generalization.

## Experiments and Running the Code
### Key Points
- The current repository runs on Python Version 3.9.16 and setuptools 65.5.0
- Run the following code to install required packages
``` 
conda create -n "rllib" python=3.9.16 -y ;\
conda activate rllib ;\
pip install wheel==0.38.4;\
pip install setuptools==65.5.0 ;\
pip install -r requirements.txt ;\
pip install -e . ;
```

To Run the script while directing the output to a log file

```
ray start --head
nohup python your_script.py > script.log 2>&1 &
```

To Restrict GPUS while running
```
export CUDA_VISIBLE_DEVICES=0,1 

nohup python scripts/run_multiagent_{algorithm_name} --args > script.log 2>&1 &
```

Additionally we can also change the resources per worke by initialize ray in the terminal

```
ray start --head --num_gpus n
```

For example in order to run RePReL we can use the scripts/run_multiagent_reprel.py with the following arguments

```
nohup python scripts/run_multiagent_reprel.py \
    --json "config/DQN_rainbow_config_v2.json" \
    --env "MultiAgentTaxi-T2-P4-v0" \
    --env-config "taxi_2T_4P_gym_default" \
    --exp-type "Transfer_2T_3P_to_4P" \
    --transfer True \
    --restore "logs/MultiAgentTaxi/RePReL/DQN_rainbow_config_v2/2T2P/T4_2024-02-05_18-14_3_2024-02-05_18-14-09/checkpoint_002977"\
    --num-workers 4\
     > MaRePReL_Taxi_Transfer_2T_3P_4P.log 2>&1 &
```

The last line can store the logs in a .log file

## Citation
Please cite the work as

