nohup python scripts/run_multiagent_reprel_dungeon.py \
    --env "MultiAgentDungeon-3A-v1" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1S" \
    --exp-type "New_3A_1S_t100" \
    --num-workers 2\
     > RePReL_Dungeon_3A_1S_t100.log 2>&1 && \
nohup python scripts/run_multiagent_reprel_dungeon.py \
    --env "MultiAgentDungeon-3A-v2" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D_1S" \
    --exp-type "New_3A_1D_1S_t100" \
    --num-workers 2\
     > RePReL_Dungeon_3A_1D_1S_t100.log 2>&1 &

