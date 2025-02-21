nohup python scripts/run_multiagent_qmix.py \
    --env 'MultiAgentOffice-2A-medium-wbump-v2' \
    --json "config/QMIX_config_office.json" \
    --env-config "office_2A_deliver_mail_coffee_med_grid_t100_h100_bump" \
    --exp-type "Transfer_2A_Deliver_Med_Bump" \
    --transfer True \
    --restore "logs/MultiAgentOffice/QMIX/QMIX_config_office/2A_Get_Med_Bump/best_ep_rew_mean_checkpoint_003000/checkpoint_003000" \
    --num-workers 2 \
     > QMIX_Transfer_Office_2A_deliver_med_bump.log 2>&1 &



