from gym.envs.registration import register
from multiagent_envs.taxi.taxi_configs import *
from multiagent_envs.officeworld.office_configs import *
from multiagent_envs.dungeon.dungeon_configs import *
# from multiagent_envs.collect_and_build.collect_and_build_configs import *

register(
    id='MultiAgentTaxi-T2-P2-v0',
    entry_point='multiagent_envs.taxi:MA_TaxiWorld',
    kwargs= taxi_2T_2P_gym_default,
)

register(
    id='MultiAgentTaxi-T2-P2-v1',
    kwargs = taxi_2T_2P_openai_default,
    entry_point='multiagent_envs.taxi:MA_TaxiWorld',
)

register(
    id='MultiAgentTaxi-T2-P2-v2',
    kwargs = taxi_2T_2P_ps_default,
    entry_point='multiagent_envs.taxi:MA_TaxiWorld',
)

register(
    id='MultiAgentTaxi-T2-P2-v3',
    kwargs = taxi_2T_2P_gym_joint,
    entry_point='multiagent_envs.taxi:MA_TaxiWorld',
)

register(
    id='MultiAgentTaxi-T2-P3-v0',
    entry_point='multiagent_envs.taxi:MA_TaxiWorld',
    kwargs = taxi_2T_3P_gym_default,
)


register(
    id='MultiAgentTaxi-T2-P3-v2',
    entry_point='multiagent_envs.taxi:MA_TaxiWorld',
    kwargs = taxi_2T_3P_ps_default,
)

register(
    id='MultiAgentTaxi-T2-P4-v0',
    entry_point='multiagent_envs.taxi:MA_TaxiWorld',
    kwargs= taxi_2T_4P_gym_default,
)


register(
    id='MultiAgentTaxi-T2-P4-v2',
    entry_point='multiagent_envs.taxi:MA_TaxiWorld',
    kwargs = taxi_2T_4P_ps_default,
)




register(
    id='MultiAgentOffice-2A-inc-terminal-rew-v0',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs =office_2A_visit_t1000_invalid0_new,
)

register(
    id='MultiAgentOffice-2A-new-grid-bump-v0',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs =office_2A_visit_new_grid_t100_h100_bump,
)

register(
    id='MultiAgentOffice-2A-new-grid-bump-v1',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs =office_2A_get_mail_coffee_new_grid_t100_h100_bump,
)

register(
    id='MultiAgentOffice-2A-new-grid-bump-v2',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs =office_2A_deliver_mail_coffee_new_grid_t100_h100_bump,
)


register(
    id='MultiAgentOffice-2A-new-grid-v0',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs =office_2A_visit_new_grid_t100_h100,
)

register(
    id='MultiAgentOffice-2A-new-grid-v1',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs =office_2A_get_mail_coffee_new_grid_t100_h100,
)

register(
    id='MultiAgentOffice-2A-new-grid-v2',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs =office_2A_deliver_mail_coffee_new_grid_t100_h100,
)


register(
    id='MultiAgentOffice-2A-default-v0',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_visit_default ,
)


register(
    id='MultiAgentOffice-2A-default-v1',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_get_mail_coffee_default ,
)



register(
    id='MultiAgentOffice-2A-default-v2',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_deliver_mail_coffee_default
)


register(
    id='MultiAgentOffice-2A-medium-v0',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_visit_med_grid_t100_h100_nobump ,
)


register(
    id='MultiAgentOffice-2A-medium-v1',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_get_mail_cofee_med_grid_t100_h100_nobump,
)



register(
    id='MultiAgentOffice-2A-medium-v2',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_deliver_mail_coffee_med_grid_t100_h100_nobump,
)


register(
    id='MultiAgentOffice-2A-medium-wbump-v0',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_visit_med_grid_t100_h100_bump ,
)


register(
    id='MultiAgentOffice-2A-medium-wbump-v1',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_get_mail_cofee_med_grid_t100_h100_bump,
)


register(
    id='MultiAgentOffice-2A-medium-wbump-v2',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_deliver_mail_coffee_med_grid_t100_h100_bump,
)


register(
    id='MultiAgentOffice-2A-medium-wbump-new-v0',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_visit_med_grid_t100_h100_bump_v2 ,
)


register(
    id='MultiAgentOffice-2A-medium-wbump-new-v1',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_get_mail_coffee_med_grid_t100_h100_bump_v2,
)


register(
    id='MultiAgentOffice-2A-medium-wbump-new-v2',
    entry_point='multiagent_envs.officeworld:MultiAgentOfficeWorld',
    kwargs = office_2A_deliver_mail_coffee_med_grid_t100_h100_bump_v2,
)

register(
    id="MultiAgentDungeon-3A-v0",
    entry_point='multiagent_envs.dungeon:DungeonEnv',
    kwargs = dungeon_3A_1D
)

register(
    id="MultiAgentDungeon-3A-v1",
    entry_point='multiagent_envs.dungeon:DungeonEnv',
    kwargs = dungeon_3A_1S
)

register(
    id="MultiAgentDungeon-3A-v2",
    entry_point='multiagent_envs.dungeon:DungeonEnv',
    kwargs = dungeon_3A_1D_1S
)

# register(
#     id = "CollectAndBuild-2A-v0",
#     entry_point='multiagent_envs.collect_and_build:MultiAgentCollectAndBuild',
#     kwargs = build_1_house
# )

# register(
#     id = "CollectAndBuild-2A-v1",
#     entry_point='multiagent_envs.collect_and_build:MultiAgentCollectAndBuild',
#     kwargs = build_1_mansion
# )

# register(
#     id = "CollectAndBuild-2A-v2",
#     entry_point='multiagent_envs.collect_and_build:MultiAgentCollectAndBuild',
#     kwargs = build_1_house_1_mansion
# )









