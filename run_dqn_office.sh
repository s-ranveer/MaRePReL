nohup python scripts/run_multiagent_dqn.py \
    --env "MultiAgentOffice-2A-default-v0" \
    --json "config/DQN_rainbow_config_v5.json" \
    --env-config "office_2A_visit_default" \
    --exp-type "2A_Visit_Default_Grid" \
    --num-workers 2\
     > DQN_Office_2A_visit_default.log 2>&1 &
PID7=$!
wait "$PID7" 


nohup python scripts/run_multiagent_dqn.py \
    --env "MultiAgentOffice-2A-default-v1" \
    --json "config/DQN_rainbow_config_v5.json" \
    --env-config "office_2A_get_mail_coffee_default" \
    --exp-type "2A_Get_Default_Grid" \
    --num-workers 2\
     > DQN_Office_2A_get_default.log 2>&1 &
PID7=$!
wait "$PID7" 



nohup python scripts/run_multiagent_dqn.py \
    --env "MultiAgentOffice-2A-default-v2" \
    --json "config/DQN_rainbow_config_v5.json" \
    --env-config "office_2A_deliver_mail_coffee_default" \
    --exp-type "2A_Deliver_Default_Grid" \
    --num-workers 2\
     > DQN_Office_2A_deliver_default.log 2>&1 &
PID7=$!
wait "$PID7" 

