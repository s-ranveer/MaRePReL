nohup python scripts/run_multiagent_dqn_with_sp.py \
    --env 'MultiAgentOffice-2A-medium-wbump-v1' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_get_mail_cofee_med_grid_t100_h100_bump" \
    --exp-type "2A_New_Get_Med_Bump_t100" \
    --num-workers 4 \
     > DQN_SP_Office_2A_get_med_bump_t100.log 2>&1 &


