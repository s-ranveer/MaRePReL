nohup python scripts/run_multiagent_dqn_transfer_pe.py \
    --json "config/DQN_rainbow_config_v7.json" \
    --env 'MultiAgentOffice-2A-medium-wbump-v1' \
    --env-config "office_2A_get_mail_cofee_med_grid_t100_h100_bump" \
    --exp-type "Transfer_2A_Get_Med_Bump" \
    --env-domain "office" \
    --restore-checkpoint "logs/MultiAgentOffice/DQN_with_SP_DQN/DQN_rainbow_config_v7/2A_New_Visit_Med_Bump_t100/best_success_checkpoint_003000/checkpoint_003000" \
     > DQN_PE_Transfer_Office_2A_Get_med_bump.log 2>&1 &
