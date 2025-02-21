nohup python scripts/run_multiagent_dqn_with_sp.py \
    --json "config/DQN_rainbow_config_v2.json" \
    --env "MultiAgentTaxi-T2-P4-v0" \
    --env-config "taxi_2T_4P_gym_default" \
    --exp-type "2T4P" \
    --num-workers 2 \
     > taxi_dqn_with_sp_2t_4p.log 2>&1 &
