nohup python scripts/run_multiagent_dqn_ps.py \
    --env 'MultiAgentOffice-2A-medium-wbump-new-v1' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_get_mail_coffee_med_grid_t100_h100_bump_v2" \
    --exp-type "2A_New_grid_Get_Med_Bump_t100" \
    --num-workers 2\
     > PS_Office_2A_get_med_bump_t100.log 2>&1 && \ 

nohup python scripts/run_multiagent_qmix.py \
    --env 'MultiAgentOffice-2A-medium-wbump-new-v1' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_get_mail_coffee_med_grid_t100_h100_bump_v2" \
    --exp-type "2A_New_grid_Get_Med_Bump_t100" \
    --num-workers 2\
     > QMIX_Office_2A_get_med_bump_t100.log 2>&1 &