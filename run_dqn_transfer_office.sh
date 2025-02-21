nohup python scripts/run_multiagent_dqn.py \
    --env 'MultiAgentOffice-2A-medium-wbump-v1' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_get_mail_cofee_med_grid_t100_h100_bump" \
    --exp-type "Transfer_2A_Get_Med_Bump" \
    --transfer True \
    --restore "logs/Done/MultiAgentOffice/IQL_DQN/DQN_rainbow_config_v7/2A_Visit_Med_Bump/best_ep_rew_mean_checkpoint_003000/checkpoint_003000" \
    --num-workers 2 \
     > IQL_Transfer_Office_2A_Get_med_bump.log 2>&1 &





