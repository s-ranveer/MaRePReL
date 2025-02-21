nohup python scripts/run_multiagent_dqn.py \
    --env 'MultiAgentOffice-2A-medium-v0' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_visit_med_grid_t100_h100_nobump" \
    --exp-type "2A_Visit_Med_NoBump" \
    --num-workers 2 \
     > DQN_Office_2A_visit_med_nobump.log 2>&1 &&
nohup python scripts/run_multiagent_dqn.py \
    --env 'MultiAgentOffice-2A-medium-v1' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_get_mail_cofee_med_grid_t100_h100_nobump" \
    --exp-type "2A_Get_Med_NoBump" \
    --num-workers 2 \
     > DQN_Office_2A_get_med_nobump.log 2>&1 && \
nohup python scripts/run_multiagent_dqn.py \
    --env 'MultiAgentOffice-2A-medium-v2' \
    --json "config/DQN_rainbow_config_v7.json" \
    --env-config "office_2A_deliver_mail_coffee_med_grid_t100_h100_nobump" \
    --exp-type "2A_Deliver_Med_NoBump" \
    --num-workers 2 \
     > DQN_Office_2A_deliver_med_nobump.log 2>&1 &


