import numpy as np

from gym import Env, spaces
from gym.utils import seeding
import pygame
import sys
from six import StringIO, b
import copy
from gym import utils
from gym.envs.toy_text import discrete
from multiagent_envs.craftworld.success_functions import *

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTIONS = [(-1, 0),  # 3,5 to 2,5 Up
           (1, 0),  # 2, 6 to 3, 6 Down
           (0, -1),  # 9 ,7 to 9.6 left
           (0, 1),  # 2, 5 to 2, 6 right
           'pickup',
           'drop',
           'exit']


WHITE = (255, 255, 255)

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


AGENT = 'agent'
PICKUPABLE = ['hammer', 'axe', 'sticks']
BLOCKING = ['rock', 'tree']

HOLDING = 'holding'
CLIP = True
OBJECTS = ['rock', 'hammer', 'tree', 'axe', 'bread', 'sticks', 'house', 'wheat']
OBJECT_PROBS = [0.25, 0.0, 0.25, 0.0, 0.1, 0.2, 0.0, 0.2]
ALL_TASKS = ['EatBread', 'GoToHouse', 'MakeBread', 'ChopTree',  'BuildHouse']
TASKS = [eval_eatbread, eval_gotohouse, eval_makebread, eval_choptree, eval_buildhouse, eval_choprock]


class CraftingBase(Env):
    """
    size=[10,10]: The size of the grid before rendering. [10,10] indicates that 
        there are 100 positions the agent can have.
    res=39: The resolution of the observation, must be a multiple of 3.
    add_objects=[]: List of objects that must be present at each reset.
    visible_agent=True: Whether to render the agent in the observation.
    state_obs=False: If true, the observation space will be the positions of the
        objects, agents, etc.
    few_obj=False: If true, only one of each object type will be spawned per episode
    use_exit=False: If true, an additional action will allow the agent to choose when
        to end the episode
    success_function=None: A function applied to two states which evaluates whether 
        success was obtained between those states.
    pretty_renderable=False: If True, env will load assets for pretty rendering.
    fixed_init_state=False: If True, env will always reset to the same state, stored in 
        self.FIXED_INIT_STATE
    """

    def __init__(self, size=[10, 10], res=3, add_objects=[], visible_agent=True, state_obs=False, state_fusion=False,
                 few_obj=False, success_reward=30,
                 use_exit=False, success_function=None, pretty_renderable=True, fixed_init_state=False, task_id=None,
                 step_cost=1.0, screen_size=1000, render_with_info = True,
                 visible_obstacle=True, visible_neighbour=False, invalid_step_cost=2.0):
        self.nrow, self.ncol = size
        self.tasks = ALL_TASKS
        self.reward_range = (0, 1)
        self.renderres = 9
        self.ACTIONS = ACTIONS
        self.visible_obstacle = visible_obstacle
        self.visible_neighbour = visible_neighbour
        self.step_cost = step_cost
        self.invalid_step_cost = invalid_step_cost
        self.success_reward = success_reward
        if not use_exit:
            self.ACTIONS = self.ACTIONS[:-1]
        nA = len(self.ACTIONS)
        nS = self.nrow * self.ncol
        self.add_objects = add_objects
        self.nS = nS
        self.nA = nA
        self.lastaction = None
        self.lastreward = None
        self.visible_agent = visible_agent
        self.few_obj = few_obj
        self.episode = 0
        self.state_obs = state_obs
        self.state_fusion = state_fusion
        self.task_id = task_id
        self.grid_dim = nS
        
        if self.task_id is None:
            self.success_function = success_function
        else:
            self.success_function = TASKS[self.task_id]

        self.action_space = spaces.Discrete(self.nA)
        self.fixed_init_state = fixed_init_state
        if self.fixed_init_state:
            self.FIXED_INIT_STATE = {
                'holding': '', 'hunger': 1.0, 'agent': (7, 8), 'count': 0,
                'object_positions': {'wheat_1': (4, 6), 'tree_1': (0, 5), 'house_1': (7, 1),
                                     'axe_1': (2, 4), 'tree_2': (8, 5), 'bread_1': (2, 0),
                                     'rock_2': (6, 2), 'hammer_1': (7, 7), 'sticks_1': (3, 4),
                                     'rock_1': (1, 0), 'rock_1': (1, 8)},
                'object_counts': {'axe': 1, 'hammer': 1, 'house': 1, 'tree': 2,
                                  'sticks': 1, 'rock': 2, 'bread': 1, 'wheat': 1}}
        if self.state_obs:
            assert (self.few_obj)
            self.max_num_per_obj = 3
            # the agent pos, objects pos , holding flag, hunger flag
            self.state_space_size = len(OBJECTS) * 2 * self.max_num_per_obj + 2 + 1 + 1
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_space_size,))
            self.state_space = self.observation_space
            self.goal_space = self.observation_space

        elif self.state_fusion:
            assert (self.few_obj)
            self.max_num_per_obj = 3
            self.image_height = self.nrow
            self.image_width = self.ncol
            self.image_channels = 1
            # the agent pos, objects pos , holding flag, hunger flag
            self.state_space_size = self.grid_dim + len(OBJECTS) * 2 * self.max_num_per_obj
            self.object_vector_size = len(OBJECTS) * 2 * self.max_num_per_obj
            if self.visible_neighbour:
                self.state_space_size += 4
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_space_size,))
        else:
            self.observation_space = spaces.Box(low=0, high=1., shape=((self.nrow + 1) * res * self.ncol * res * 3,))
        self.objects = []
        self.res = res
        self.screen_size = screen_size
        self.render_with_info = render_with_info
        self.height, self.width = screen_size,screen_size
        if(pretty_renderable):
            pygame.init()
            self.pretty_render_res = self.screen_size // self.nrow 
            if(render_with_info):
                self.surf = pygame.display.set_mode((self.width, int(self.height*1.2)))   
            else:
                self.surf = pygame.display.set_mode((self.width, int(self.height)))   
            pygame.display.set_caption('Craftworld')


        self._init_sprites(pretty_renderable)

    def _init_sprites(self, pretty_renderable):
        sprites = []
        if CLIP:
            base_sprite = np.zeros((3, 3, 3)) / 10.
        else:
            base_sprite = np.ones((3, 3, 3)) / 10.
        for i in range(3):
            for j in range(3):
                new_sprite = base_sprite.copy()
                new_sprite[i, j, :] = 1.0
                sprites.append(new_sprite)

        SPRITES = {'agent': sprites[0],
                   'rock': sprites[1],
                   'hammer': sprites[2],
                   'tree': sprites[3],
                   'bread': sprites[4],
                   'wheat': sprites[5],
                   'sticks': sprites[6],
                   'axe': sprites[7],
                   'house': sprites[8],
                   # 'wheat': sprites[9],
                   }
        # BGR
        SPRITES['agent'][0, 0] = np.array([0 / 255., 0 / 255., 255 / 255.])
        SPRITES['rock'][0, 1] = np.array([211 / 255., 211 / 255., 211 / 255.])
        SPRITES['hammer'][0, 2] = np.array([204 / 255., 204 / 255., 0 / 255.])
        SPRITES['tree'][1, 0] = np.array([34 / 255., 133 / 255., 34 / 255.])
        SPRITES['bread'][1, 1] = np.array([0 / 255., 215 / 255., 255 / 255.])
        SPRITES['wheat'][1, 2] = np.array([10 / 255., 215 / 255., 100 / 255.])
        SPRITES['sticks'][2, 0] = np.array([45 / 255., 82 / 255., 160 / 255.])
        SPRITES['axe'][2, 1] = np.array([255 / 255., 102 / 255., 102 / 255.])
        SPRITES['house'][2, 2] = np.array([153 / 255., 52 / 255., 255 / 255.])
        BIGSPRITES = copy.deepcopy(SPRITES)
        self.SPRITES = SPRITES
        self.BIGSPRITES = BIGSPRITES
        for obj in self.SPRITES.keys():
            size = self.SPRITES[obj].shape[0]
            if size < self.res:
                new_sprite = np.repeat(self.SPRITES[obj] * 255, repeats=self.res / size, axis=1)
                new_sprite = np.repeat(new_sprite, repeats=self.res / size, axis=0)
                SPRITES[obj] = new_sprite / 255
            size = self.BIGSPRITES[obj].shape[0]

            if size < self.renderres:
                new_sprite = np.repeat(self.BIGSPRITES[obj] * 255, repeats=self.renderres / size, axis=1)
                new_sprite = np.repeat(new_sprite, repeats=self.renderres / size, axis=0)
                self.BIGSPRITES[obj] = new_sprite / 255

        if pretty_renderable:
            import os
            
        
            self.render_order = ['house', 'tree', 'rock', 'sticks', 'wheat', 'hammer', 'axe', 'bread', 'agent']
            asset_path = '/'.join(os.path.realpath(__file__).split('/')[:-1] + ['assets/*.png'])
            print("asset_path", asset_path)
            import glob
            import cv2
            asset_paths = glob.glob(asset_path)

            self.pretty_render_sprites = {asset.split('/')[-1].split('.')[0]: cv2.imread(asset) for asset in
                                          asset_paths}

    def sample_objects(self):
        num_objects = np.random.randint(15, 25)
        indices = np.random.multinomial(1, OBJECT_PROBS, size=num_objects)
        indices = np.argmax(indices, axis=1)
        for obj in OBJECTS:
            i = 1
            self.objects.append(obj)
        if not self.few_obj:
            for i in range(max(num_objects - len(self.objects), 0)):
                obj_idx = indices[i]
                obj = OBJECTS[obj_idx]
                self.objects.append(obj)
        return self.objects

    def from_s(self, s):
        row = int(s / self.ncol)
        return (row, s - row * self.ncol)

    def to_s(self, row, col):
        return row * self.ncol + col

    def get_root(self, obj):
        if obj is None or obj == HOLDING or obj == 'hunger' or obj == '':
            return None
        elif '_' in obj:
            return obj.split('_')[0]
        elif obj == 'agent':
            return obj

    def _reset_from_init(self, init_from_state=None):
        """ 
        init_from_state: If a dictionary state is passed here, the environment
        will reset to that state. Otherwise, the state is randomly initialized.
        """
        self.viewer = None
        if self.fixed_init_state:
            init_from_state = copy.deepcopy(self.FIXED_INIT_STATE)
        if init_from_state is None:
            self.init_from_state = False
            self.state = {}
            self.state[HOLDING] = ''
            self.state['hunger'] = 1.0
            self.state['count'] = 0

            self.objects = []
            self.sample_objects()

            self.objects += self.add_objects
            self.object_counts = {k: 0 for k in OBJECTS}
            self.obj_max_index = copy.deepcopy(self.object_counts)
            self.state['obj_max_index'] = self.obj_max_index
            self.state['object_counts'] = self.object_counts
            self.state['object_positions'] = {}

            positions = np.random.permutation(self.nS)[:len(self.objects) + 2]
            agent_pos = self.from_s(positions[0])
            self.state['agent'] = agent_pos
            j = 1
            for i, ob in enumerate(self.objects):
                pos = self.from_s(positions[i + j])
                if np.sum(pos) == 0:
                    j += 1
                    pos = self.from_s(positions[i + j])
                self._add_obj(ob, pos)

        else:
            self.init_from_state = True
            self.state = init_from_state
            self.object_counts = self.state['object_counts']
            self.state['obj_max_index'] = copy.deepcopy(self.object_counts)
            self.obj_max_index = self.state['obj_max_index']
            self.objects = [obj for obj in self.object_counts.keys()]
        self.lastaction = None
        self.episode_reward = 0
        total = self.verify_env()
        self.total_count = total
        self.init_state = copy.deepcopy(self.state)
        self.episode_states = [self.init_state]

    def _step_internal(self, a):
        total = self.verify_env()
        prev_state = copy.deepcopy(self.state)
        self.state['count'] += 1
        assert (total <= self.total_count)
        self.total_count = total
        r = -self.step_cost
        action = ACTIONS[a]
        # if action == 'exit':
        #     d = True
        if action == 'pickup':
            self.try_pickup()
        elif action == 'drop':
            self.try_drop()
        else:
            old_pos = copy.deepcopy(self.state['agent'])
            new_pos = self.move_agent(a)
            if new_pos == old_pos:
                r -= self.invalid_step_cost
        self.lastaction = a
        
        success = 0
        self.episode_states.append(copy.deepcopy(self.state))
        if self.success_function is not None:
            success = self.success_function(prev_state, self.state)
            if success:
                r += self.success_reward
        self.lastreward =r
        self.episode_reward += r
        return r, success, {'success': success, 'count': self.state['count'], 'episode reward': self.episode_reward}

    def get_diagnostics(self, paths, **kwargs):
        successes = [p['env_infos'][-1]['success'] for p in paths]
        success_rate = sum(successes) / len(successes)
        lengths = [p['env_infos'][-1]['success'] for p in paths]
        length_rate = sum(lengths) / len(lengths)
        rewards = [p['env_infos'][-1]['episode reward'] for p in paths]
        average_reward = np.mean(rewards)
        reward_max = np.max(rewards)
        return {'Success Rate': success_rate,
                'Episode length Mean': length_rate,
                'Episode length Min': min(lengths),
                'Episode counts': len(paths),
                'Total Reward Mean': average_reward,
                'Total Reward Max': reward_max}

    def _sample_free_square(self):
        perm = np.random.permutation(self.nS)
        for s in perm:
            pos = self.from_s(s)
            if pos not in self.state.values():
                return pos

    def _add_obj(self, objtype, pos):
        if objtype not in self.object_counts:
            self.object_counts[objtype] = 0
        suffix = self.obj_max_index[objtype] + 1
        self.obj_max_index[objtype] += 1
        self.object_counts[objtype] += 1
        self.state['object_positions'][objtype + '_' + str(suffix)] = pos

    def _remove_obj(self, obj):
        objtype = obj.split('_')[0]
        if objtype not in self.object_counts:
            import pdb;
            pdb.set_trace()
        self.object_counts[objtype] -= 1
        del self.state['object_positions'][obj]

    def _perform_object(self, obj):
        blocked = False
        if obj.startswith('tree'):
            if self.state[HOLDING].startswith('axe'):
                pos = self.state['object_positions'][obj]
                self._add_obj('sticks', pos)
                self._remove_obj(obj)
            else:
                blocked = True
        elif obj.startswith('rock'):
            if self.state[HOLDING].startswith('hammer'):
                self._remove_obj(obj)
            else:
                blocked = True
        elif obj.startswith('bread') or obj.startswith('house'):
            self.state['hunger'] = 0
            if obj.startswith('bread'):
                self._remove_obj(obj)
        elif obj.startswith('sticks') and 'hammer' in self.state[HOLDING]:
            pos = self.state['object_positions'][obj]
            self._add_obj('house', pos)
            self._remove_obj(obj)
        elif obj.startswith('wheat') and 'axe' in self.state[HOLDING]:
            pos = self.state['object_positions'][obj]
            self._add_obj('bread', pos)
            self._remove_obj(obj)
        return blocked

    def move_agent(self, a):
        act = ACTIONS[a]
        pos = self.state[AGENT]
        row, col = pos[0] + act[0], pos[1] + act[1]
        if row in range(self.nrow) and col in range(self.ncol):
            local_objects = []
            for obj in self.state['object_positions'].keys():
                root = self.get_root(obj)
                if root != 'agent':
                    obj_pos = self.state['object_positions'][obj]
                    if obj_pos == (row, col):
                        local_objects.append(obj)
            is_blocked = False
            for obj in local_objects:
                blocked = self._perform_object(obj)
                is_blocked = blocked or is_blocked
            # Check obstacles:
            if is_blocked:
                return pos

            self.state[AGENT] = (row, col)
            if len(self.state[HOLDING]) > 0:
                obj = self.state[HOLDING]
                self.state['object_positions'][obj] = (row, col)
            return (row, col)
        else:
            return pos

    def try_pickup(self):
        pos = self.state[AGENT]
        for obj in self.state['object_positions'].keys():
            root = self.get_root(obj)
            if root is not None and root != obj:
                obj_pos = self.state['object_positions'][obj]
                if obj_pos == pos and root in PICKUPABLE:
                    if self.state[HOLDING] == '':
                        self.state[HOLDING] = obj
        return

    def try_drop(self):
        # Can only drop if nothing else is there
        pos = self.state[AGENT]
        if self.state[HOLDING] is not None:
            for obj in self.state['object_positions'].keys():
                root = self.get_root(obj)
                if root is not None and root != obj:
                    obj_pos = self.state['object_positions'][obj]
                    if obj_pos == pos and obj != self.state[HOLDING]:
                        return
            self.state[HOLDING] = ''
        return

    def verify_env(self):
        """ Check that the number of objects in the state is what we expect."""
        my_obj_counts = {k: 0 for k in OBJECTS}
        for obj in self.state['object_positions'].keys():
            if obj != 'agent' and obj != 'holding' and obj != 'object_counts' and obj != 'hunger':
                objtype = obj.split('_')[0]
                if objtype not in my_obj_counts:
                    my_obj_counts[objtype] = 0
                my_obj_counts[objtype] += 1
        for k in my_obj_counts.keys():
            if my_obj_counts[k] != self.object_counts[k]:
                import pdb;
                pdb.set_trace()
            assert (my_obj_counts[k] == self.object_counts[k])
        for k in self.object_counts.keys():
            assert (my_obj_counts[k] == self.object_counts[k])
        return sum(my_obj_counts.values())

    def get_env_obs(self):
        if self.state_obs:
            obs = self.imagine_obs(self.state)
            obs = obs / self.nS
            return obs
        elif self.state_fusion:
            return self.imagine_obs(self.state)
        else:
            img = np.zeros(((self.nrow + 1) * self.res, self.ncol * self.res, 3))
            to_get_obs = self.state['object_positions'].keys()
            for obj in to_get_obs:
                root = self.get_root(obj)
                if root is not None:
                    if root in self.SPRITES:
                        row, col = self.state['object_positions'][obj]
                        img[row * self.res:(row + 1) * self.res, col * self.res:(col + 1) * self.res, :] += \
                        self.SPRITES[root]

            if self.visible_agent:
                row, col = self.state[AGENT]
                img[row * self.res:(row + 1) * self.res, col * self.res:(col + 1) * self.res, :] += self.SPRITES[AGENT]
            w, h, c = img.shape
            img[w - self.res:w, 0:self.res, 0] = self.state['hunger']
            img[w - self.res:w, self.res:self.res * 2, :] = (len(self.state[HOLDING]) > 0)
            if CLIP:
                img = np.clip(img, 0, 1.0)
            output = img.flatten() * 255
            output = output.astype(np.uint8)
            return output

    def get_obs(self):
        return self.get_env_obs()

    def imagine_obs(self, state, mode='rgb'):
        if self.state_obs:
            obs = np.zeros(self.state_space_size)
            obs[:2] = state['agent']
            for obj, pos in state['object_positions'].items():
                root, num = obj.split('_')
                num = int(num) - 1
                assert (num < 3)
                assert (num >= 0)
                idx = OBJECTS.index(root) * 2 * self.max_num_per_obj + 2 + num * 2
                obs[idx:idx + 2] = pos

            obs[-2] = state['hunger']
            obs[-1] = (len(state[HOLDING]) > 0)

            return obs
        if self.state_fusion:
            obs = []
            array_pos = np.zeros((self.nrow, self.ncol))
            vector_obj = np.zeros(3 * 2 * len(OBJECTS))

            agent_state = state["agent"]
            if (state["hunger"]):
                array_pos[agent_state[0], agent_state[1]] = 0.7
            else:
                array_pos[agent_state[0], agent_state[1]] = 0.3

            if self.visible_neighbour:
                neighbours = [(agent_state[0] + ACTIONS[a][0], agent_state[1] + ACTIONS[a][1]) for a in range(4)]
                neighbour_blocked = [0., 0., 0., 0.]

            for obj, pos in state['object_positions'].items():
                if self.visible_obstacle:
                    if obj.startswith('rock') or obj.startswith('tree'):
                        array_pos[pos[0], pos[1]] = 1.0
                if self.visible_neighbour and pos in neighbours:
                    if (obj.startswith('rock') and not self.state[HOLDING].startswith('hammer')) or \
                            (obj.startswith('tree') and not self.state[HOLDING].startswith('axe')):
                        neighbour_blocked[neighbours.index(pos)] = 1.
                root, num = obj.split('_')
                num = int(num) - 1
                assert (num < 3)
                assert (num >= 0)
                idx = OBJECTS.index(root) * 2 * self.max_num_per_obj + num * 2
                vector_obj[idx:idx + 2] = pos

            vector_pos = np.stack(array_pos, axis=0).ravel()
            # for obj, pos in state['object_positions'].items():
            #     vector_obj[vector_idx] = pos[0]/100
            #     vector_obj[vector_idx+1] = pd1
            # root, num = obj.split('_')
            # num = int(num)-1
            # assert(num <3)
            # assert(num >=0)
            # idx = OBJECTS.index(root)*2*self.max_num_per_obj+2+num*2
            # obs[idx:idx+2] = pos

            vector_obj /= 10
            if (self.state[HOLDING]):
                obj = self.state[HOLDING]
                root, num = obj.split('_')
                num = int(num) - 1
                assert (num < 3)
                assert (num >= 0)
                idx = OBJECTS.index(root) * 2 * self.max_num_per_obj + num * 2
                vector_obj[idx:idx + 2] = (-1., -1.)
            if self.visible_neighbour:
                vector_obj = np.append(vector_obj, neighbour_blocked)
            obs = np.append(vector_pos, vector_obj)
            # return vector_obj
            return obs

        if mode == 'rgb':
            img = np.zeros(((self.nrow + 1) * self.res, self.ncol * self.res, 3))
            to_get_obs = state['object_positions'].keys()
            for obj in to_get_obs:
                root = self.get_root(obj)
                if root is not None:
                    if root in self.SPRITES:
                        row, col = state['object_positions'][obj]
                        img[row * self.res:(row + 1) * self.res, col * self.res:(col + 1) * self.res, :] += \
                        self.SPRITES[root]

            if self.visible_agent:
                row, col = state[AGENT]
                img[row * self.res:(row + 1) * self.res, col * self.res:(col + 1) * self.res, :] += self.SPRITES[AGENT]

            w, h, c = img.shape
            img[w - self.res:w, 0:self.res, 0] = self.state['hunger']
            img[w - self.res:w, self.res:self.res * 2, :] = (len(self.state[HOLDING]) > 0)
            return img.flatten()

    def center_agent(self, img, res):
        new_obs = np.zeros((img.shape[0] * 2, img.shape[1] * 2, 3)) + 0.1
        row, col = self.state[AGENT]
        disp_x = img.shape[0] - row * res
        disp_y = img.shape[1] - col * res
        new_obs[disp_x:disp_x + img.shape[0], disp_y:disp_y + img.shape[1]] = img
        return new_obs

    def render_env(self, mode='rgb', width=800, height=800):
        if mode == 'human':
            return self.pretty_render(mode)
        else:
            # Create a Pygame surface
            cell_width = width // (self.nrow + 1)
            cell_height = height // self.ncol
            
            # Define colors based on BIGSPRITES
            colors = {
                'agent': self.BIGSPRITES[AGENT],
                'hunger': (self.state['hunger'], 0, 0),  # Assuming hunger is represented by a red component
                'holding': (255, 255, 255) if len(self.state[HOLDING]) > 0 else (0, 0, 0),  # White if holding, else black
            }
            
            img = pygame.Surface((width, height))
            
            to_get_obs = self.state['object_positions'].keys()
            for obj in to_get_obs:
                root = self.get_root(obj)
                if root is not None and root in self.SPRITES:
                    row, col = self.state['object_positions'][obj]
                    pygame.draw.rect(img, self.BIGSPRITES[root], (col * cell_width, row * cell_height, cell_width, cell_height))

            if self.visible_agent:
                row, col = self.state[AGENT]
                pygame.draw.rect(img, colors['agent'], (col * cell_width, row * cell_height, cell_width, cell_height))

            w, h, c = img.get_width(), img.get_height(), 3
            img_array = pygame.surfarray.array3d(img)

            img_array[w - cell_height:w, 0:cell_width, 0] = colors['hunger'][0]
            img_array[w - cell_height:w, cell_width:cell_width * 2, :] = colors['holding']

            img = pygame.surfarray.make_surface(img_array)
            
            return img

    def resize_pretty_render_sprites(self, width, height):
            # Assuming pretty_render_sprites is a dictionary containing sprite surfaces
            # Iterate through the dictionary and resize each sprite
            for key in self.pretty_render_sprites:
                original_surface = self.pretty_render_sprites[key]
                surface = pygame.surfarray.make_surface(original_surface)
                resized_surface = pygame.transform.scale(surface, (width, height))
                self.pretty_render_sprites[key] = pygame.surfarray.array3d(resized_surface)

    def pretty_render(self, mode='human'):
        
        cell_width = self.screen_size // (self.nrow)
        cell_height = self.screen_size // (self.ncol)
 
        img = np.zeros(((self.nrow) * self.pretty_render_res, self.ncol * self.pretty_render_res, 3)).astype(np.uint8)
       
        self.resize_pretty_render_sprites(width=cell_width, height=cell_height)
        grass = (self.pretty_render_sprites['grass2'] / 3).astype(np.uint8)
        for row in range(self.nrow):
            for col in range(self.ncol):
                img[row * self.pretty_render_res:(row + 1) * self.pretty_render_res,
                        col * self.pretty_render_res:(col + 1) * self.pretty_render_res] = grass
       

        to_get_obs = self.state['object_positions'].keys()
        for to_render_obj in self.render_order:
            if to_render_obj == 'agent':
                sprite = self.pretty_render_sprites[to_render_obj]
                row, col = self.state[AGENT]
                gray_pixels = np.max(sprite, axis=2)
                idx = np.where(gray_pixels > 0)
                col_offset = col * self.pretty_render_res
                row_offset = row * self.pretty_render_res
                img[(idx[0] + row_offset, idx[1] + col_offset)] = sprite[idx]
            else:
                for obj in to_get_obs:
                    root = self.get_root(obj)
                    if root == to_render_obj:
                        sprite = self.pretty_render_sprites[to_render_obj]
                        row, col = self.state['object_positions'][obj]
                        gray_pixels = np.max(sprite, axis=2)
                        idx = np.where(gray_pixels > 0)
                        col_offset = col * self.pretty_render_res
                        row_offset = row * self.pretty_render_res
                        img[(idx[0] + row_offset, idx[1] + col_offset)] = sprite[idx]

        w, h, c = img.shape
        if len(self.state[HOLDING]) > 0:
            root = self.get_root(self.state[HOLDING])
            img[w - self.pretty_render_res:w, self.pretty_render_res:self.pretty_render_res * 2] = self.pretty_render_sprites[root]

        img = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
        return img

    def check_move_agent(self, a):
        """ Does not actually change state"""
        act = ACTIONS[a]

        pos = self.state[AGENT]
        row, col = pos[0] + act[0], pos[1] + act[1]
        # Check bounds
        removes_obj = None
        blocked = False
        if row in range(self.nrow) and col in range(self.ncol):
            local_objects = []
            for obj in self.state['object_positions'].keys():
                root = self.get_root(obj)
                if root != 'agent':
                    obj_pos = self.state['object_positions'][obj]
                    if obj_pos == (row, col):
                        local_objects.append(obj)
            is_blocked = False
            for obj in local_objects:
                blocked = False
                if obj.startswith('tree'):
                    if not self.state[HOLDING].startswith('axe'):
                        blocked = True
                    else:
                        removes_obj = 'tree'
                elif obj.startswith('rock'):
                    if not self.state[HOLDING].startswith('hammer'):
                        blocked = True
                    else:
                        removes_obj = 'rock'
                elif obj.startswith('bread'):
                    removes_obj = 'bread'
                elif obj.startswith('wheat') and self.state[HOLDING].startswith('axe'):
                    removes_obj = 'wheat'
        else:
            blocked = True
        if blocked:
            row, col = pos
        return (row, col), blocked, removes_obj

    ################################################################################
    # Goal generation functions

    def _add_obj_to_state(self, state, objtype, pos):
        if objtype not in state['object_counts']:
            state['object_counts'][objtype] = 0
        suffix = state['obj_max_index'][objtype] + 1
        state['obj_max_index'][objtype] += 1
        state['object_counts'][objtype] += 1
        state['object_positions'][objtype + '_' + str(suffix)] = pos

    def _remove_obj_from_state(self, state, obj):
        objtype = obj.split('_')[0]
        if objtype not in state['object_counts']:
            import pdb;
            pdb.set_trace()
        state['object_counts'][objtype] -= 1
        del state['object_positions'][obj]

    def get_objects_by_type(self, name, state):
        objects = []
        for obj in state['object_positions'].keys():
            if obj.startswith(name):
                objects.append(obj)
        return objects

    def get_random_object_by_type(self, name, state=None):
        if state is None:
            state = self.state
        objects = self.get_objects_by_type(name, state)
        obj = objects[np.random.randint(0, len(objects))]
        return obj

    def generate_goal_state(self, task_id, init_state):
        goal_state = copy.deepcopy(init_state)
        if task_id == 0:  # EatBread
            if init_state['object_counts']['bread'] > 0:
                bread = self.get_random_object_by_type('bread', state=goal_state)
                pos = goal_state['object_positions'][bread]
                goal_state['hunger'] = 0.0
                goal_state['agent'] = pos
                self._remove_obj_from_state(goal_state, bread)
        elif task_id == 1:  # GoToHouse
            if init_state['object_counts']['house'] > 0:
                house = self.get_random_object_by_type('house', state=goal_state)
                pos = goal_state['object_positions'][house]
                goal_state['agent'] = pos
        elif task_id == 2:  # ChopRock
            if init_state['object_counts']['rock'] > 0 and init_state['object_counts']['hammer'] > 0:
                rock = self.get_random_object_by_type('rock', state=goal_state)
                pos = goal_state['object_positions'][rock]
                self._remove_obj_from_state(goal_state, rock)
                hammer = self.get_random_object_by_type('hammer')
                goal_state['agent'] = pos
                goal_state['object_positions'][hammer] = pos
                goal_state['holding'] = hammer
        elif task_id == 3:  # ChopTree
            if init_state['object_counts']['tree'] > 0 and init_state['object_counts']['axe'] > 0:
                tree = self.get_random_object_by_type('tree', state=goal_state)
                pos = goal_state['object_positions'][tree]
                self._remove_obj_from_state(goal_state, tree)
                self._add_obj_to_state(goal_state, 'sticks', pos)
                axe = self.get_random_object_by_type('axe')
                goal_state['agent'] = pos
                goal_state['object_positions'][axe] = pos
                goal_state['holding'] = axe
        elif task_id == 4:  # BuildHouse
            if init_state['object_counts']['sticks'] > 0 and init_state['object_counts']['hammer'] > 0:
                sticks = self.get_random_object_by_type('sticks', state=goal_state)
                pos = goal_state['object_positions'][sticks]
                self._remove_obj_from_state(goal_state, sticks)
                self._add_obj_to_state(goal_state, 'house', pos)
                hammer = self.get_random_object_by_type('hammer')
                goal_state['agent'] = pos
                goal_state['object_positions'][hammer] = pos
                goal_state['holding'] = hammer
        elif task_id == 5:  # MakeBread
            if init_state['object_counts']['wheat'] > 0 and init_state['object_counts']['axe'] > 0:
                wheat = self.get_random_object_by_type('wheat', state=goal_state)
                pos = goal_state['object_positions'][wheat]
                self._remove_obj_from_state(goal_state, wheat)
                self._add_obj_to_state(goal_state, 'bread', pos)
                axe = self.get_random_object_by_type('axe')
                goal_state['agent'] = pos
                goal_state['object_positions'][axe] = pos
                goal_state['holding'] = axe
        elif task_id == 6:  # MoveAxe
            if init_state['object_counts']['axe'] > 0:
                obj = self.get_random_object_by_type('axe', state=goal_state)
                newpos = self.from_s(np.random.randint(0, self.nS))
                goal_state['object_positions'][obj] = newpos
                goal_state['holding'] = obj
                goal_state['agent'] = newpos
        elif task_id == 7:  # MoveHammer
            if init_state['object_counts']['hammer'] > 0:
                obj = self.get_random_object_by_type('hammer', state=goal_state)
                newpos = self.from_s(np.random.randint(0, self.nS))
                goal_state['object_positions'][obj] = newpos
                goal_state['holding'] = obj
                goal_state['agent'] = newpos
        elif task_id == 8:  # MoveSticks
            if init_state['object_counts']['sticks'] > 0:
                obj = self.get_random_object_by_type('sticks', state=goal_state)
                newpos = self.from_s(np.random.randint(0, self.nS))
                goal_state['object_positions'][obj] = newpos
                goal_state['holding'] = obj
                goal_state['agent'] = newpos
        elif task_id == 9:  # GoToPosition
            newpos = self.from_s(np.random.randint(0, self.nS))
            goal_state['agent'] = newpos
        return goal_state

    def get_objects_from_obs(self, obs, obj):
        poses = []
        if self.state_obs or self.state_fusion:
            for num in range(self.max_num_per_obj):
                idx = OBJECTS.index(obj) * 2 * self.max_num_per_obj + 2 + num * 2
                if obs[idx] > 0 or obs[idx + 1] > 0:
                    poses.append((obs[idx], obs[idx + 1]))
        else:
            raise NotImplementedError()
        return poses

    def eval_tasks(self, init_obs, final_obs):
        self.tasks = ['EatBread', 'GoToHouse', 'ChopRock', 'ChopTree', 'BuildHouse', 'MakeBread', 'MoveAxe',
                      'MoveHammer', "MoveSticks"]
        task_success = {}
        if self.state_obs:
            init_objects = {obj: self.get_objects_from_obs(init_obs, obj) for obj in OBJECTS}
            final_objects = {obj: self.get_objects_from_obs(final_obs, obj) for obj in OBJECTS}
            task_success['MakeBread'] = len(final_objects['wheat']) < len(init_objects['wheat'])
            task_success['EatBread'] = (len(final_objects['bread']) + len(final_objects['wheat'])) < (
                    len(init_objects['bread']) + len(init_objects['wheat']))
            task_success['BuildHouse'] = len(final_objects['house']) > len(init_objects['house'])
            task_success['ChopTree'] = len(final_objects['tree']) < len(init_objects['tree'])
            task_success['ChopRock'] = len(final_objects['rock']) < len(init_objects['rock'])
            task_success['GoToHouse'] = (final_obs[0], final_obs[1]) in final_objects['house']
            task_success['MoveAxe'] = final_objects['axe'] != init_objects['axe']
            task_success['MoveHammer'] = final_objects['hammer'] != init_objects['hammer']
            task_success['MoveSticks'] = False in [stick in init_objects['sticks'] for stick in final_objects['sticks']]
            task_list = [task_success[key] for key in self.tasks]
            return np.array(task_list)
        else:
            raise NotImplementedError()

    def add_text(self,text, text_type): 
        size = self.screen_size
        if(text_type=="title"):  
            title_font_style = "liberationmono"   
            font_size_title = int(size*0.04)   
            title_font = pygame.font.Font(pygame.font.match_font(title_font_style), font_size_title)
            output= title_font.render(text, True, WHITE)
        
        elif(text_type=="body"):    
            game_font_style = "dejavusans"
            font_size_game_info = int(size*0.018)    
            game_font = pygame.font.Font(pygame.font.match_font(game_font_style), font_size_game_info)
            output= game_font.render(text, True, WHITE)

        return output