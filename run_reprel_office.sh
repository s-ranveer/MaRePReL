nohup python scripts/run_multiagent_reprel.py \
    --env "MultiAgentOffice-2A-default-v0" \
    --json "config/DQN_rainbow_config_v9.json" \
    --env-config "office_2A_visit_default" \
    --exp-type "2A_Visit_Default_Grid_new_abs_low_gamma_high_lr" \
    --num-workers 2\
     > RePReL_Office_2A_visit_default_low_g_high_lr.log 2>&1 && \ 
     nohup python scripts/run_multiagent_reprel.py \
    --env "MultiAgentOffice-2A-default-v1" \
    --json "config/DQN_rainbow_config_v9.json" \
    --env-config "office_2A_get_mail_coffee_default" \
    --exp-type "2A_Get_Default_Grid_new_abs_low_gamma_high_lr" \
    --num-workers 2\
     > RePReL_Office_2A_get_default_low_g_high_lr.log 2>&1 && \
     nohup python scripts/run_multiagent_reprel.py \
    --env "MultiAgentOffice-2A-default-v2" \
    --json "config/DQN_rainbow_config_v9.json" \
    --env-config "office_2A_deliver_mail_coffee_default" \
    --exp-type "2A_Deliver_Default_Grid_new_abs_low_gamma_high_lr" \
    --num-workers 2\
     > RePReL_Office_2A_deliver_default_low_g_high_lr.log 2>&1 &
