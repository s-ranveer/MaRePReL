dungeon_3A_1D = {
    "t_config": {
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "horizon": 500,
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 1,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 0,
            "attack": 1,
            "hp": 10,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 0,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}

dungeon_3A_1S = {
    "t_config": {
    "horizon": 500,
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 0,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 1,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 0,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}

dungeon_3A_1W = {
    "t_config": {
    "horizon": 500,
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 0,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 0,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 1,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}
dungeon_3A_1D_1S = {
    "t_config": {
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "horizon": 1000,
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 1,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 1,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 0,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}
dungeon_3A_2D = {
    "t_config": {
    "horizon": 500,
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 2,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 0,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 0,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}
dungeon_3A_2S = {
    "horizon": 500,
    "t_config": {
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 0,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 2,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 0,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}

dungeon_3A_2W = {
    "t_config": {
    "horizon": 500,
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 0,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 0,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 2,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}
dungeon_3A_3D = {
    "t_config": {
    "horizon": 500,
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 2,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 0,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 0,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}

dungeon_3A_3S = {
    "t_config": {
    "horizon": 500,
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 0,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 3,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 0,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}

dungeon_3A_3W = {
    "t_config": {
    "horizon": 500,
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 0,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 0,
            "attack": 1,
            "hp": 10,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 3,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}

dungeon_3A_1D_1S_1W = {
    "t_config": {
        "horizon": 500,
    "dungeon_layout": [
            ["X", "X", "X", "D", "X",   "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X",  " ", " ", " ", " ",  "X"],
            ["X", " ",  " ", " ", " ",  "X"],
            ["X",  "P", " ", " ", "P",  "X"],
            ["X", "X",  "X", "X", "X",  "X"],
        ],
    "render": "human",
    "max_players": 3,
    "player_attack": 1,
    "max_player_health": 3,
    "friendly_fire": False,
    "rewards": {
        "enabled": False,
        "attack_enemy": 3,
        "attack_agent": -3,
        "kill_enemy": 10,
        "kill_agent": -10,
        "pickup": 5,
        "unlock_key": 10,
        "unlock_door": 50,
        "failure": -30
    },
    "enemies": {
        "dragon": {
            "count": 1,
            "attack": 2,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "skeleton": {
            "count": 1,
            "attack": 0.5,
            "hp": 5,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        },
        "wraith": {
            "count": 1,
            "attack": 3,
            "hp": 3,
            "agitated": -1,
            "time_to_attack": 1,
            "key": True
        }
    }
}
}