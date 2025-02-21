nohup python scripts/run_multiagent_dqn_dungeon_ps.py \
    --env "MultiAgentDungeon-3A-v2" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D_1S" \
    --exp-type "Transfer_New_3A_1D_to_1D_1S" \
    --transfer "True" \
    --restore "logs/MultiAgentDungeon/PS_DQN/DQN_rainbow_config_dunge/New_3A_1D/best_ep_rew_mean_checkpoint_003000/checkpoint_003000" \
    --num-workers 2 \
     > Transfer_DQN_PS_Dungeon_3A_1D_to_1D_1S.log 2>&1 & 
