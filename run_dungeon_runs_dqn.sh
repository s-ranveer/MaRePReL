

nohup python scripts/run_multiagent_dqn_ps.py \
    --env "MultiAgentDungeon-3A-v0" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D" \
    --exp-type "3A_1D" \
    --num-workers 2\
     > DQN_PS_Dungeon_3A_1D.log 2>&1 &
PID8=$!
wait "$PID8" 

nohup python scripts/run_multiagent_dqn_ps.py \
    --env "MultiAgentDungeon-3A-v1" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1S" \
    --exp-type "3A_1S" \
    --num-workers 2\
     > DQN_PS_Dungeon_3A_1S.log 2>&1 &
PID8=$!
wait "$PID8" 

nohup python scripts/run_multiagent_dqn_ps.py \
    --env "MultiAgentDungeon-3A-v2" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D_1S" \
    --exp-type "3A_1D_1S" \
    --num-workers 2\
     > DQN_PS_Dungeon_3A_1D_1S.log 2>&1 &
PID8=$!
wait "$PID8" 
