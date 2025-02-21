from multiagent_envs.collect_and_build.env_descriptors import *

build_1_house = {
    "t_config": {
        "terrain": [["X", " ", " ", " ", " ", " ", "X"],
                    [" ", "X", " ", " ", " ", "X", " "],
                    [" ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", " ", " ", " ", " ", " "],
                    [" ", "X", " ", " ", " ", "X", " "],
                    ["X", " ", " ", " ", " ", " ", "X"]],
        "resource_tiles": [wood_medium, wood_medium, stone_small, stone_medium],
        "n_agents": 2,
        "to_build": [house],
        "horizon": 500,
        "step_cost": -0.1,
        "terminal_reward": 100
    }
}

build_1_mansion = {
    "t_config": {
        "terrain": [["X", "X", " ", " ", " ", " ", " ", "X", "X"],
                    ["X", " ", " ", "X", " ", "X", " ", " ", "X"],
                    [" ", " ", "X", " ", " ", " ", "X", " ", " "],
                    [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", " ", " ", " ", " ", " ", " " ," "],
                    [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", "X", " ", " ", " ", "X", " ", " "],
                    ["X", " ", " ", "X", " ", "X", " ", " ", "X"],
                    ["X", "X", " ", " ", " ", " ", " ", "X", "X"]],
        "resource_tiles": [wood_medium, wood_medium, stone_small, stone_medium],
        "n_agents": 2,
        "to_build": [mansion],
        "horizon": 500,
        "step_cost": -0.1,
        "terminal_reward": 100
    }
}

build_1_house_1_mansion = {
    "t_config": {
        "terrain": [["X", "X", " ", " ", " ", " ", " ", "X", "X"],
                    ["X", " ", " ", "X", " ", "X", " ", " ", "X"],
                    [" ", " ", "X", " ", " ", " ", "X", " ", " "],
                    [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                    [" ", " ", "X", " ", " ", " ", "X", " ", " "],
                    ["X", " ", " ", "X", " ", "X", " ", " ", "X"],
                    ["X", "X", " ", " ", " ", " ", " ", "X", "X"]],
        "resource_tiles": [wood_medium, wood_medium, stone_small, stone_medium],
        "n_agents": 2,
        "to_build": [house, mansion],
        "horizon": 500,
        "step_cost": -0.1,
        "terminal_reward": 100
    }
}