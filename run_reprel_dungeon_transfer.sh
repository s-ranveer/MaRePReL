nohup python scripts/run_multiagent_reprel_dungeon.py \
    --transfer "True" \
    --env "MultiAgentDungeon-3A-v0" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D" \
    --exp-type "Transfer_3A_1S_to_1D_t100" \
    --num-workers 2\
    --restore "logs/MultiAgentDungeon/RePReL/DQN_rainbow_config_dunge/New_3A_1S_t100/best_ep_rew_mean_checkpoint_003000/checkpoint_003000" \
     > transfer_RePReL_Dungeon_3A_1S_to_1D_t100.log 2>&1 &
