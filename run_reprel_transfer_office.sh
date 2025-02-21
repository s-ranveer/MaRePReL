
nohup python scripts/run_multiagent_reprel.py \
    --env 'MultiAgentOffice-2A-medium-wbump-new-v2' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_deliver_mail_coffee_med_grid_t100_h100_bump_v2" \
    --exp-type "Transfer_2A_New_grid_Deliver_Med_Bump_t100" \
    --transfer True \
    --restore "logs/MultiAgentOffice/RePReL/DQN_rainbow_config_v7/2A_New_grid_Get_Med_Bump_t100/best_success_checkpoint_003000/checkpoint_003000" \
    --num-workers 4\
    > RePReL_Transfer_New_grid_Office_2A_deliver_med_bump_t100.log 2>&1 &


