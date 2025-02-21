nohup python scripts/run_multiagent_dqn_transfer_pe.py \
    --env "MultiAgentDungeon-3A-v2" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D_1S" \
    --env-domain "dungeon" \
    --restore-checkpoint "logs/MultiAgentDungeon/DQN_with_SP_DQN/DQN_rainbow_config_dunge/3A_1D/T1_2024-10-14_12-07_0_2024-10-14_12-07-55/checkpoint_003000" \
     > DQN_PE_Transfer_Dungeon_1D_1S.log 2>&1 &
