nohup python scripts/run_multiagent_reprel.py \
    --env 'MultiAgentOffice-2A-medium-wbump-new-v2' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_deliver_mail_coffee_med_grid_t100_h100_bump_v2" \
    --exp-type "2A_New_grid_Deliver_Med_Bump_t100" \
    --num-workers 2\
     > RePReL_Office_new_grid_2A_deliver_med_bump_t100.log 2>&1 &



