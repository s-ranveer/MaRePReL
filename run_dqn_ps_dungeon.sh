nohup python scripts/run_multiagent_dqn_dungeon_ps.py \
    --env "MultiAgentDungeon-3A-v2" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D_1S" \
    --exp-type "New_3A_1D_1S" \
    --num-workers 2\
     > DQN_PS_Dungeon_3A_1D_1S.log 2>&1 &
