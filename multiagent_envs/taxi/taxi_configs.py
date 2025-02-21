from multiagent_envs.taxi.taxiworld_gen import *

taxi_2T_2P_gym_default = {"t_config":
    {"passenger_count" : 2,
    "taxi_count": 2, 
    "format":"decentralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"gym", 
    "random_seed": 0, 
    "max_steps":200,
    "agent_ids":None,
    "actions_are_logits": False,
    "description": "default observation type (gym) with 2 passengers and 2 taxis"}
    }

taxi_2T_2P_gym_joint = {"t_config":
    {"passenger_count" : 2,
    "taxi_count": 2, 
    "format":"centralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"gym", 
    "random_seed": 0, 
    "max_steps":200,
    "agent_ids":None,
    "actions_are_logits": False,
    "description": "centralized observation type (gym) with 2 passengers and 2 taxis"}
    }


taxi_2T_2P_openai_default = {"t_config":
    {"passenger_count" : 2,
    "taxi_count": 2, 
    "format":"decentralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"openai", 
    "random_seed": 42, 
    "max_steps":200,
    "description": "default observation type (openai) with 2 passengers and 2 taxis"}
    }


taxi_2T_2P_gym_wstate = {"t_config":
    {"passenger_count" : 2,
    "taxi_count": 2, 
    "format":"decentralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"gym",
    "with_state": True, 
    "random_seed": 42, 
    "max_steps":200,
    "description": "default observation type (openai) with 2 passengers and 2 taxis"}
    }


taxi_2T_2P_ps_default = {"t_config":
    {"passenger_count" : 2,
    "taxi_count": 2, 
    "format":"decentralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"full_parameter_sharing", 
    "random_seed": 42, 
    "max_steps":200,
    "description": "default observation type (parameter sharing) with 2 passengers and 2 taxis"}
    }


taxi_2T_3P_gym_default={"t_config":
    {"passenger_count" : 3,
    "taxi_count": 2, 
    "format":"decentralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"gym", 
    "random_seed": 42, 
    "max_steps":200}
    }

taxi_2T_3P_ps_default = {"t_config":
    {"passenger_count" : 3,
    "taxi_count": 2, 
    "format":"decentralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"full_parameter_sharing", 
    "random_seed": 42, 
    "max_steps":200,
    "description": "default observation type (parameter sharing) with 3 passengers and 2 taxis"}
    }


taxi_2T_4P_gym_default={"t_config":
    {"passenger_count" : 4,
    "taxi_count": 2, 
    "format":"decentralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"gym", 
    "random_seed": 42, 
    "max_steps":200}
}

taxi_2T_4P_ps_default = {"t_config":
    {"passenger_count" : 4,
    "taxi_count": 2, 
    "format":"decentralized", 
    "reward_drop":20, 
    "step_cost":-0.1, 
    "crash_penalty":-100, 
    "no_move_cost":-1,
    "reward_pickup":20, 
    "max_passenger":4, 
    "random_pickup_drop":-1, 
    "layout":default_layout, 
    "obs_type":"full_parameter_sharing", 
    "random_seed": 42, 
    "max_steps":200,
    "description": "default observation type (parameter sharing) with 4 passengers and 2 taxis"}
    }
