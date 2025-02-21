nohup python scripts/run_multiagent_dqn_ps.py \
    --env 'MultiAgentOffice-2A-medium-wbump-v2' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_deliver_mail_coffee_med_grid_t100_h100_bump" \
    --exp-type "2A_Deliver_Med_Bump" \
    --num-workers 2 \
     > DQN_PS_Office_2A_deliver_med_bump.log 2>&1 &


