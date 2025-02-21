
nohup python scripts/run_multiagent_dqn_dungeon.py \
    --transfer "True" \
    --env "MultiAgentDungeon-3A-v2" \
    --env-config "dungeon_3A_1D_1S" \
    --exp-type "Transfer_3A_1D_to_1D_1S" \
    --restore "logs/MultiAgentDungeon/IQL_DQN/DQN_rainbow_config_dunge/New_3A_1D/best_episode_reward_mean_checkpoint_3000" \
     > transfer_IQL_3A_1D_to_1D_1S_t100.log 2>&1 && \
nohup python scripts/run_multiagent_dqn_dungeon.py \
    --transfer "True" \
    --env "MultiAgentDungeon-3A-v0" \
    --json "config/DQN_rainbow_config_dungeon.json" \
    --env-config "dungeon_3A_1D" \
    --exp-type "Transfer_3A_1S_to_1D" \
    --num-workers 2\
    --restore "logs/MultiAgentDungeon/IQL_DQN/DQN_rainbow_config_dunge/New_3A_1S/best_ep_rew_mean_checkpoint_3000" \
     > transfer_IQL_Dungeon_3A_1D_to_1S_t100.log 2>&1 &
