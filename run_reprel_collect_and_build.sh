

nohup python scripts/run_multiagent_reprel_collectbuild.py \
    --env "CollectAndBuild-2A-v1" \
    --json "config/DQN_rainbow_config_collectbuild.json" \
    --env-config "build_1_mansion" \
    --exp-type "MaRePReL_test_2" \
    --num-workers 2\
     > RePReL_CollectBuild_1M_new_config_test.log 2>&1 &
PID10=$!
wait "$PID10" 

# nohup python scripts/run_multiagent_reprel_dungeon.py \
#     --env "MultiAgentDungeon-3A-v1" \
#     --json "config/DQN_rainbow_config_dungeon.json" \
#     --env-config "dungeon_3A_1S" \
#     --exp-type "New_3A_1S_t10" \
#     --num-workers 2\
#      > RePReL_Dungeon_3A_1S_t10.log 2>&1 &
# PID10=$!
# wait "$PID10" 

# nohup python scripts/run_multiagent_reprel_dungeon.py \
#     --env "MultiAgentDungeon-3A-v2" \
#     --json "config/DQN_rainbow_config_dungeon.json" \
#     --env-config "dungeon_3A_1D_1S" \
#     --exp-type "New_3A_1D_1S_t10" \
#     --num-workers 2\
#      > RePReL_Dungeon_3A_1D_1S_t10.log 2>&1 &
# PID10=$!
# wait "$PID10" 
