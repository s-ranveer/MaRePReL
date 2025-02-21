nohup python scripts/run_multiagent_reprel_dungeon.py \
    --env 'MultiAgentDungeon-3A-v2' \
    --env-config "dungeon_3A_1D_1S" \
    --exp-type "MaRePReL_3A_1D_1S" \
     > MaRePReL_3A_1D_1S_2.log 2>&1 
PID4=$!
wait "$PID4"