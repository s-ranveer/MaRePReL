nohup python scripts/run_multiagent_qmix.py \
    --env 'MultiAgentOffice-2A-medium-wbump-v0' \
    --json "config/QMIX_config_office.json" \
    --env-config "office_2A_visit_med_grid_t100_h100_bump" \
    --exp-type "2A_Visit_Med_Bump" \
    --num-workers 2\
     > QMIX_Office_2A_visit_med_bump.log 2>&1 &&
nohup python scripts/run_multiagent_qmix.py \
    --env 'MultiAgentOffice-2A-medium-wbump-v1' \
    --json "config/QMIX_config_office.json" \
    --env-config "office_2A_get_mail_cofee_med_grid_t100_h100_bump" \
    --exp-type "2A_Get_Med_Bump" \
    --num-workers 2\
     > QMIX_Office_2A_get_med_bump.log 2>&1 && \
nohup python scripts/run_multiagent_qmix.py \
    --env 'MultiAgentOffice-2A-medium-wbump-v2' \
    --json "config/QMIX_config_office.json" \
    --env-config "office_2A_deliver_mail_coffee_med_grid_t100_h100_bump" \
    --exp-type "2A_Deliver_Med_Bump" \
    --num-workers 2\
     > QMIX_Office_2A_deliver_med_bump.log 2>&1 &



