from utils.aima_planner import *


st = taxi_domain()

pop = PartialOrderPlanner(st)

pop.execute()