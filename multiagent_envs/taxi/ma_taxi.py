from PIL import Image, ImageDraw, ImageFont

#Ray IMPORTS
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from ray.rllib.env.env_context import EnvContext

#Taxi Imports
from multiagent_envs.taxi.taxiworld_gen import *

#Gym Imports
from gym.spaces.discrete import Discrete
from gym.spaces import Box, Dict

from utils.utils import *

#numpy,os
from collections import Counter
import numpy as np
import os


ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
    4: 'pick up',
    5: 'drop'
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

class MA_TaxiWorld(MultiAgentEnv):
    def __init__(self, t_config:EnvContext):
        #ASSERTIONS
        assert t_config.get("passenger_count") <=  t_config.get("max_passenger")
        assert t_config.get("taxi_count")<=3
        self._skip_env_checking = True

        #DESCRIPTION()[0].
        self._description = t_config.get("Description")

        # Global State Variable (T/F)
        self.with_state = t_config.get("with_state", False)
        
        # ESSENTIALS
        agent_names = []
        self.actions_are_logits = t_config.get("actions_are_logits")
        if t_config.get("agent_ids") is None:
            for i in range(0, t_config.get("taxi_count")):
                agent = "AGENT" + str(i + 1)
                agent_names.append(agent)
        
        else:
            agent_names = t_config.get("agent_ids")
            
        self.passenger_count = t_config.get("passenger_count")
        self.max_passenger = t_config.get("max_passenger")
        self.taxi_count = t_config.get("taxi_count") #UPDATED FOR MA
        self.n_agents = self.taxi_count
        self.random_seed = t_config.get("random_seed")
        
            # except:
            #     self.random_seed = t_config.get("random_seed")


        self.format = t_config.get("format")
        self.layout = t_config.get("layout")
        
        #VECTORIZED VARIABLES
        self.action_space = Discrete(len(ACTION_LOOKUP)) # UPDATED FOR MA

        # VECTORIZED VARIABLES
        self.action_space = Discrete(len(ACTION_LOOKUP))  # UPDATED FOR MA

        # MAPPING FOR TAXI
        self.agent_ids = agent_names
        self.agent_ids = set(self.agent_ids)
        self._agent_ids = list(set(agent_names))  # NECESSARY FOR THE WRAPPER CLASS
        self.taxi_ids = mapping(agent_names, [i for i in range(self.taxi_count)])  # UPDATED FOR MA
        
        # print(self.taxi_ids)

        self.target_diagnostics = ["passengers_picked" , "passengers_dropped", "episode_reward", "episode_length", "success", "crash", "total_tasks"]
        self.target_totals = ["total_taxis", "total_tasks"]
        self.user_data_fields = self.target_diagnostics

        #REWARD/PENALTY VARIABLES
        self.reward_drop = t_config.get("reward_drop")
        self.step_cost = t_config.get("step_cost")
        self.no_move_cost = t_config.get("no_move_cost")
        self.reward_pickup = t_config.get("reward_pickup")
        self.max_steps = t_config.get("max_steps")
        self.crash_penalty = t_config.get("crash_penalty")
        
        #FOR VISUALIZATON 
        self.obs_type = t_config.get("obs_type")

        #UNKNOWN VARIABLES
        self.random_pickup_drop = t_config.get("random_pickup_drop")
               
        
        #LAYOUT AND WALLS
        self.layout = np.array([list(line) for line in self.layout.splitlines()])
        self.grid_width = len(self.layout[0])
        wall = self.layout == 'w'
        self.not_wall = ~wall  #

        # CANDIDATE LOCATION VARIABLES FOR RESET
        self.locations = to_xy(self.not_wall.nonzero())
        self.passenger_candidates = [tuple(np.argwhere(self.layout == x)[0]) for x in ['R', 'G', 'B', 'Y']]
        self.passenger_RGBY = dict(zip(["R","G","B","Y"],self.passenger_candidates))
        self.taxi_candidates = to_xy((self.layout == ' ').nonzero())
        
        #INFO FOR THE OBSERVATION VARIABLE
        obs = self.reset()
        if self.format == "decentralized":
            self.obs_size = len(obs[list(self.agent_ids)[0]])
        else:
            self.obs_size = len(obs)
            
        if self.obs_type == "gym" or self.obs_type == "openai" or self.obs_type == "full_parameter_sharing":

            if self.format == "decentralized":

                if self.with_state:
                    self.observation_space = Dict({"obs":Box(-1, 1, shape=tuple([self.obs_size+1]), dtype='float32'), ENV_STATE: Box(-1, 1, shape=tuple([self.obs_size]), dtype='float32')})
                
                else:
                    self.observation_space = Box(0, 1, shape=tuple([self.obs_size]), dtype='float32')
                    self.observation_space_dict = {x:Box(0, 1, shape=tuple([self.obs_size]), dtype='float32') for x in self.agent_ids}
                    
                self.action_space_dict = {x: Discrete(len(ACTION_LOOKUP)) for x in obs.keys()}

            else:
                if self.obs_type =="full_parameter_sharing":
                    print("Full Paramater Sharing has to be decentralized")
                    exit()
                
                if self.with_state:
                    self.observation_space =  Dict({"obs":Box(-1, 1, shape=tuple([self.obs_size+1]), dtype='float32'), ENV_STATE: Box(-1, 1, shape=tuple([self.obs_size]), dtype='float32')})
                else:
                    self.observation_space = Box(0, 1, shape=tuple([self.obs_size]), dtype='float32')

            map = (self.not_wall == 0).astype(float)
            map[self.passenger_candidates[0]] = 101
            map[self.passenger_candidates[1]] = 102
            map[self.passenger_candidates[2]] = 103
            map[self.passenger_candidates[3]] = 104
            map[tuple(self.taxi_locs.values())] = 0.5

            image = map[1:-1, 1:-1]

            grid_vector = np.stack(image, axis=0).ravel()
            self.grid_dim = len(grid_vector)
            self.image_height, self.image_width = image.shape
            self.image_channels = 1

            if self.obs_type == "graph":
                self.obj_dim = 5
            else:
                self.obj_dim = 9
                
            self.object_vector_size = self.obj_dim * self.max_passenger
            
            
    @property
    def get_agent_ids(self):
        return self._agent_ids
    
    @property
    def description(self):
        return self._description
    
    @property
    def observation(self):
            if "openai" == self.obs_type or "gym"==self.obs_type or "full_parameter_sharing"==self.obs_type:
                grid = np.array(list(self.taxi_locs.values())) / 10

            for i, (pick, drop) in enumerate(zip(self.pickup, self.drop)):
                pickup_vector = np.zeros(len(self.passenger_candidates) + 1)
                drop_vector = np.zeros(len(self.passenger_candidates))
                if not self.passenger_done[i]:
                    passengers = list(self.taxi_passenger.values())
                    if i in passengers:
                        
                        pickup_vector[-1] = (passengers.index(i) + 1 )/10
                        #Updated it to show which taxi picked up the passenger (1 for AGENT1; 2 for AGENT2)
                    else:
                        pickup_vector[self.passenger_candidates.index(pick)] = 1
                    drop_vector[self.passenger_candidates.index(drop)] = 1
                grid = np.append(np.append(grid, pickup_vector.copy()), drop_vector.copy())

            if self.passenger_count < self.max_passenger:
                grid = np.append(grid, np.zeros(9 * (self.max_passenger - self.passenger_count)))


            if "gym" == self.obs_type:
                if self.format == "decentralized":
                    _obs = [grid for _ in range(self.taxi_count)]
                    if self.with_state:
                        ma_obs = {}
                        for index, i in enumerate(self.agent_ids):
                            cur_obs = _obs[0]
                            f = (index+1)/10
                            qmix_obs  = np.append(cur_obs, f) 
                            ma_obs[i] = {"obs":qmix_obs, ENV_STATE: _obs[0]}
                            
                    else:
                        ma_obs = mapping(self.agent_ids, _obs)
                    return ma_obs
                return grid

            if "openai" == self.obs_type:
                if self.format == "decentralized":
                    ma_obs = {}
                    for index, i in enumerate(self.agent_ids):
                        
                        index = self.agent_ids.index(i)

                        i_taxi_loc = list(self.taxi_locs.values())[index]
                        new_grid = list(i_taxi_loc)
                    
                        for j,loc in enumerate(self.taxi_locs.values()):
                            if(j!= index):
                            
                                diff = (i_taxi_loc[0]-loc[0], i_taxi_loc[1]-loc[1])
                                new_grid.extend(list(diff))
                
                        new_grid = np.array(new_grid)/10
                        new_grid = np.append(new_grid, grid[self.taxi_count*2:-1])
                        
                        if self.with_state:
                            cur_obs = _obs[0]
                            f = (index+1)/10
                            qmix_obs  = np.append(cur_obs, f) 
                            ma_obs[i] = {"obs":qmix_obs, ENV_STATE: _obs[0]}
                        else:
                            ma_obs[i]=np.array(new_grid)

                    return ma_obs
                
                else:
                    print("openai needs to use decentralized format")
                    quit()
            
            if "full_parameter_sharing" == self.obs_type:
                if self.format == "decentralized":
                    ma_obs = {}
                    for i in self.agent_ids:
                        
                        index = self.agent_ids.index(i)

                        taxi_locs = list(self.taxi_locs.values())
                        i_taxi_loc = taxi_locs[index]
                        new_grid = list(i_taxi_loc)
                    
                        for j,loc in enumerate(self.taxi_locs.values()):
                            if(j!= index):
                                new_grid.extend(loc)
    
                        new_grid = np.array(new_grid)/10
                        new_grid = np.append(new_grid, grid[self.taxi_count*2:-1])
                        ma_obs[i]=np.array(new_grid)

                    return ma_obs
                
                else:
                    print("openai needs to use decentralized format")
                    quit()




    
    def reset(self):
        
        #Setting Random Seed to Initialize the same env state at every reset
        np.random.seed(int(self.random_seed))
        # print("SEED: ",self.random_seed)
        
        
        """
        Passengers Pickup and drop locations (Number Min 2, Max 4)
        """
        
        #List for Pickup and Drop Location of Passengers

        #For every passenger, choose pickup and drop Location 
        ## MUST TEST
        
        self.pickup = []
        self.drop = []
        pick_loc_indices = [i for i in range(0,len(self.passenger_candidates))]
        
        for _ in range(self.passenger_count):
            pi = np.random.choice(pick_loc_indices, 1)[0]
            drop_loc_indices = [i for i in range(0,len(self.passenger_candidates))]
            drop_loc_indices.remove(pi)
            di = np.random.choice(drop_loc_indices, 1)[0]
            
            self.pickup.append(self.passenger_candidates[pi])
            pick_loc_indices.remove(pi)
            self.drop.append(self.passenger_candidates[di])
        
        self.pick_render = self.pickup.copy()
        self.drop_render = self.drop.copy()
        
        #Choosing n Random Locations from Taxi Candidates / n is the num_of_taxis
        locs = np.random.choice(len(self.taxi_candidates), self.taxi_count,replace=False) 
        
        #Dictionary mapping from Taxi name to Location
        self.taxi_locs = mapping(self.agent_ids, [self.taxi_candidates[i] for i in locs])
        
        
        #Dict for current passenger the Taxi is Carrying / Default {"Taxi_Name":None, "Taxi_Name2":None}
        self.taxi_passenger = mapping(self.agent_ids, [None for _ in range(self.taxi_count)] ) 
        
        #Dict for number of passengers each taxi picked / Default {"Taxi_Name":0, "Taxi_Name2":0}
        self.taxi_num_passengers_picked = mapping(self.agent_ids, np.zeros(self.taxi_count)) 

        #Dict for number of passengers each taxi dropped / Default {"Taxi_Name":0, "Taxi_Name2":0}
        self.taxi_num_passengers_dropped = mapping(self.agent_ids, np.zeros(self.taxi_count)) 
        
        #List for Done Condition of each passenger / Default [False, False]
        self.passenger_done = np.zeros(self.passenger_count, dtype=bool)
       
        #Dictionary for agent dones (Necessary for mutiagent) 
        # Default {"Taxi_ID1":False, "Taxi_ID2":False, "__all__": False}
        self.dones = mapping(self.agent_ids,self.passenger_done) #ONLY WORKS IN THE CASE THAT #TAXI = #PASSENGER
        self.dones["__all__"]=False
        self.any_crash = False #For Info on if any taxi crashed   
        self.hc_mapping = mapping(self.agent_ids,[False for _ in range(self.taxi_count)]) #Mapping for taxi to crash (true/false)
        self.viewer = None

        self.num_env_steps = 0
        self.episode_reward = 0

        #Dictionary for individual agent reward / Default {"Taxi_Name":0, "Taxi_Name2":0}
        self.taxi_episode_return = mapping(self.agent_ids, [self.episode_reward for _ in range(self.taxi_count)])
        
        return self.observation
        
        
    def set_seed(self,seed, verbose=False):
        if(verbose):
            print("Seed Set to : ",seed)
        self.random_seed= seed


    def is_valid_location(self, taxi):  # UPDATED FOR MA
        try:
            wall_flag = 0
            if not self.not_wall[tuple(taxi)]:
                wall_flag = True

            return not wall_flag
        except IndexError:
            return False

    def has_crashed(self, actions):  # NEW HAS CRASHED FUNCTION (UPDATED CRASH FUNCTION TO ACCOMODATE MORE THAN 2 TAXIS)
        prev_loc = []
        curr_loc = []
        crash = [False for i in range(self.taxi_count)]
        for (taxi_id, action) in sorted(actions.items()):
            if (action < 4):
                prev_loc.append(tuple(self.taxi_locs[taxi_id] - np.array(CHANGE_COORDINATES[action])))
            else:
                prev_loc.append(tuple(self.taxi_locs[taxi_id]))
            curr_loc.append(self.taxi_locs[taxi_id])
            
        prev_curr = mapping(prev_loc,curr_loc)
        
        element_counts = dict(Counter(curr_loc))
        
        crash_index = []
        for location, counts in element_counts.items():
            if counts>1:
                crash_index.extend([index for index, value in enumerate(curr_loc) if value==location])
        
        for index in crash_index:
            crash[index] = True
        
        
        for i in prev_loc:
            if prev_curr[i] in prev_loc:
                if prev_curr[prev_curr[i]] == i and prev_curr[i] != i:
                    crash[prev_loc.index(i)] = True
                    crash[curr_loc.index(i)] = True


        return crash
    
    

    def step(self, actions):  # EDITED FOR MA -> ACTION IS NOW A DICTIONARY
        self.num_env_steps += 1
        step_reward_total = 0
        taxi_reward = mapping(self.agent_ids,[self.step_cost for _ in range(self.taxi_count)])
        
        #If actions are logits (probabilities)
        
        if(self.actions_are_logits):
            actions = {k: np.argmax(v) for k,v in actions.items()}
            
        for (taxi_id, action) in actions.items():
            #Check if it is valid
            assert action in ACTION_LOOKUP.keys()
            
            # Movement Actions
            if action < 4:
                
                new_taxi = tuple(self.taxi_locs[taxi_id] + np.array(CHANGE_COORDINATES[action]))        
                if self.is_valid_location(new_taxi):
                    self.taxi_locs[taxi_id] = new_taxi
                else:
                    taxi_reward[taxi_id] += self.no_move_cost

            # Non-Movement Actions
            elif action == 4 and self.taxi_passenger[taxi_id] is None:
                # Pickup
                
                try:
                    i = self.pickup.index((tuple(self.taxi_locs[taxi_id])))
                except:
                    i = -1
                if i != -1 and not self.passenger_done[i]:
                    self.taxi_passenger[taxi_id] = i
                    self.pickup[i] = (999, 999)  # making the pickup inaccessible
                    taxi_reward[taxi_id] += self.reward_pickup
                    self.taxi_num_passengers_picked[taxi_id]+=1
                else:
                    taxi_reward[taxi_id] += self.random_pickup_drop
                    
                
                
            elif action == 5 and self.taxi_passenger[taxi_id] is not None:
                # Drop

                if self.drop[self.taxi_passenger[taxi_id]] == tuple(self.taxi_locs[taxi_id]):
                    self.passenger_done[self.taxi_passenger[taxi_id]] = True
                    self.drop[self.taxi_passenger[taxi_id]] = (999,999) # making the drop inaccessible
                    taxi_reward[taxi_id] += self.reward_drop
                    self.taxi_passenger[taxi_id] = None
                    self.taxi_num_passengers_dropped[taxi_id] += 1

                else:
                    taxi_reward[taxi_id] += self.random_pickup_drop


            else:
                # raise AssertionError("action is not valid")
                if action in (4, 5):
                    taxi_reward[taxi_id] += self.random_pickup_drop
        


        success = np.sum(self.passenger_done)==self.passenger_count #Success Condition
        hc = self.has_crashed(actions) #Has Crashed Condition (Now a vector )
        hc_mapping = mapping(self.taxi_ids, hc)
        self.hc_mapping = hc_mapping
        
                
        if any(hc):
            self.any_crash=True
            for taxi_id, crashed in hc_mapping.items():
                if(crashed):
                    taxi_reward[taxi_id] += self.crash_penalty    
                    self.dones[taxi_id] = True
            self.dones["__all__"] = True
        
        if self.num_env_steps >= self.max_steps or success:
            for taxi_id in actions.keys():
                self.dones[taxi_id] = True
            self.dones["__all__"] = True

        for taxi_id in actions.keys():
            self.taxi_episode_return[taxi_id] += taxi_reward[taxi_id] #Episode Reward Individual Taxi
            step_reward_total += taxi_reward[taxi_id]
            
        self.episode_reward += step_reward_total #Episode Reward Both Taxis combined
        self.taxi_reward = taxi_reward

        info_all = {
            "episode_length": self.num_env_steps,
            "passengers_picked" : np.sum(list(self.taxi_num_passengers_picked.values())),
            "passengers_dropped": np.sum(self.passenger_done),
            "is_success": success,
            "crash": any(hc), # ADDED A HAS CRASHED ELEMENT
            "episode_reward": self.episode_reward,
        }

        if self.format=="decentralized":
            taxi_infos = {} #{"__all__":info_all}
            for i in self.dones.keys() :
                    if(i!="__all__"):
                        info = {
                            "episode_length": self.num_env_steps,
                            "passengers_picked": self.taxi_num_passengers_picked[i],
                            "passengers_dropped": self.taxi_num_passengers_dropped[i],
                            "crashed": self.hc_mapping[i], # ADDED A HAS CRASHED ELEMENT
                            "step_reward": self.taxi_reward[i],
                            "agent_return": self.taxi_episode_return[i],
                        }
                        taxi_infos[i] = info

            return self.observation, self.taxi_reward, self.dones, taxi_infos

        else:
            return self.observation, step_reward_total, self.dones["__all__"], info_all


    def get_diagnostics(self):
        return {
                'episode_length': self.num_env_steps,
                'passengers_picked': np.sum(list(self.taxi_num_passengers_picked.values())),
                'passengers_dropped': np.sum(self.passenger_done),
                'success': np.sum(self.passenger_done)==self.passenger_count,
                'crash': self.any_crash,
                'episode_reward': self.episode_reward,
                'total_taxis': self.taxi_count,
                'total_tasks': self.passenger_count,
                }



    def get_diagnostics_ma(self):
        agent_infos ={}
        for i in self.dones.keys():
            
            if(i!="__all__"):
                info = {
                        'Passengers Picked': self.taxi_num_passengers_picked[i],
                        'Passengers Dropped': self.taxi_num_passengers_dropped[i],
                        'Crash': self.hc_mapping[i],
                        'Episode Return': self.taxi_episode_return[i],
                }
                agent_infos[i] = info
            
        return agent_infos


    def render(self, mode="save", saveloc="imgs", filename= "out", title=None, info=None):
        M, N = self.not_wall.shape
        img = np.zeros(shape=(M, N, 3), dtype=np.uint8)
        img[self.not_wall] = 255

        # for i, loc in enumerate(self.passenger_candidates):
        #     img[tuple(loc)] = RGBY[i]

        for i, taxi_loc in enumerate(self.taxi_locs.values()):

            img[tuple(taxi_loc)] = TAXI_COLOR[i]
            passengers = list(self.taxi_passenger.values())
            if(passengers[i] is not None):
                pick_loc = self.pick_render[passengers[i]]
                pick_letter = list(filter(lambda x: self.passenger_RGBY[x] == pick_loc, self.passenger_RGBY))[0]
                
                img[tuple(taxi_loc)] = PICKUP_COLOR[pick_letter]
              

        for letter in self.passenger_RGBY:
            x,y = self.passenger_RGBY[letter]
            if(self.passenger_RGBY[letter] in self.pickup):
                    img[tuple([x,y])] = PICKUP_COLOR[letter]

            if(self.passenger_RGBY[letter] in self.drop):
                    pick_loc = (self.pick_render[self.drop_render.index(self.passenger_RGBY[letter])] )
                    pick_letter = list(filter(lambda x: self.passenger_RGBY[x] == pick_loc, self.passenger_RGBY))[0]
                    if letter=="R" or letter== "G":
                        img[tuple([x-1,y])] = DROP_COLOR[pick_letter]
                    if letter=="B" or letter== "Y":
                        img[tuple([x+1,y])] = DROP_COLOR[pick_letter]  


       

        from gym.envs.classic_control import rendering  # Only works on gym==0.21

        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        scale = 16
        scaled = np.zeros((img.shape[0] * scale, img.shape[1] * scale, img.shape[2]))
        scaled[:, :, 0] = np.kron(img[:, :, 0], np.ones((scale, scale)))
        scaled[:, :, 1] = np.kron(img[:, :, 1], np.ones((scale, scale)))
        scaled[:, :, 2] = np.kron(img[:, :, 2], np.ones((scale, scale)))

        self.viewer.imshow(scaled.astype(np.uint8))

        if(mode=='save'):
            img = Image.fromarray(scaled.astype(np.uint8), 'RGB')
            width,height = img.size
            
            
            # info = {
            #         "Step Num":"unk",
            #         "Ep Num":"unk",
            #         "Ep Reward":"unk",
            #     }


            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 24)
            if(not title):
                title = "Sample Experiment"
                
            if(info):
                new = Image.new('RGBA',(width+230,height+40),'white')
                new.paste(img,(0,0,(width),(height)))
                draw = ImageDraw.Draw(new)
                draw.text((10,height+10),title,font=font,fill="black")
                font_side_text = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14)
                
                for i,(k,v) in enumerate(info.items()):
                    text = "{}: {}".format(k,v)
                
                    _, h =font.getsize(text)
                    draw.text((width+10, 10+h*i), text, font=font_side_text, fill="black")

            else:
                w, h = font.getsize(title)
                new = Image.new('RGBA',(width+w-160,height+h+15),'white')
                new.paste(img,(int((width+w-160)/5.5),5))
                draw = ImageDraw.Draw(new)
                draw.text((10,height+10),title,font=font,fill="black")

            isexist = os.path.exists(saveloc)
            if not isexist:
                os.makedirs(saveloc)
            new.save('{}/{}.png'.format(saveloc,filename))
        
        return self.viewer.isopen


if __name__ == '__main__':

    # multiagent_envs = MA_TaxiWorld( passenger_count=2, max_passenger=4)
    # env = MA_TaxiWorld(passenger_count=2, max_passenger=4, format="centralized")

    # multiagent_envs.close()

    env = MA_TaxiWorld()
    obs = env.reset()
    print(obs)