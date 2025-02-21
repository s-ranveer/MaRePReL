nohup python scripts/run_multiagent_reprel.py \
    --json "config/DQN_rainbow_config_v2.json" \
    --env "MultiAgentTaxi-T2-P4-v0" \
    --env-config "taxi_2T_4P_gym_default" \
    --exp-type "Transfer_2T_3P_to_4P" \
    --transfer True \
    --restore "logs/MultiAgentTaxi/RePReL/DQN_rainbow_config_v2/2T2P/T4_2024-02-05_18-14_3_2024-02-05_18-14-09/checkpoint_002977"\
    --num-workers 4\
     > MaRePReL_Taxi_Transfer_2T_3P_4P.log 2>&1 &


