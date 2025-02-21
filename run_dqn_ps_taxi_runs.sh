nohup python scripts/run_multiagent_dqn_ps.py \
    --json "config/DQN_rainbow_config_v2.json" \
    --env "MultiAgentTaxi-T2-P3-v0" \
    --env-config "taxi_2T_3P_gym_default" \
    --exp-type "2T3P" \
    --num-workers 2\
     > PS_Taxi_2T_3P.log 2>&1 && \
nohup python scripts/run_multiagent_dqn_ps.py \
    --json "config/DQN_rainbow_config_v2.json" \
    --env "MultiAgentTaxi-T2-P4-v0" \
    --env-config "taxi_2T_4P_gym_default" \
    --exp-type "Transfer_2T_3P_to_4P" \
    --transfer True \
    --restore "logs/MultiAgentTaxi/PS_DQN/DQN_rainbow_config_v2/2T3P/best_ep_rew_mean_checkpoint_003000" \
    --num-workers 2\
     > PS_Taxi_Transfer_2T_3P_4P.log 2>&1 &


