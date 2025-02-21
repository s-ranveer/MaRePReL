nohup python scripts/run_multiagent_dqn_with_sp.py \
    --env "MultiAgentDungeon-3A-v1" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D" \
    --exp-type "3A_1D" \
    --num-workers 2\
    > DQN_with_SP_Dungeon_3A_1D.log 2>&1 &