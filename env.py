"""Auxilary files for those who wanted to solve with policy gradient"""
import gym
from gym.core import Wrapper
from gym.spaces.box import Box
from gym.spaces import Discrete

# from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import PolycraftGym

from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np
import time

class EnvHandler:
    def __init__(self, env, n_envs):
        self.env = env
        self.observation = self.env.reset()
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.envs = [self.make_env(game)[0] for _ in range(n_envs)]

    
    def render_rgb(self):
        """ render rgb environment """
        plt.imshow(self.env.render('rgb_array'))
        plt.show()
    
    def render_frames(self, states):
        """ render processed frames """
        plt.imshow(states.transpose([0,2,1]).reshape([42,-1]))
        plt.show()

    # For forward run
    def run(self):
        """ run current environment """
        self.env.render(mode='human')
    
    # For parallel run
    def parallel_reset(self):
        """ Reset all games and return [n_envs, *obs_shape] observations """
        return np.array([env.reset() for env in self.envs])
    
    def parallel_step(self, actions):
        """
        Send a vector[batch_size] of actions into respective environments
        :returns: observations[n_envs, *obs_shape], rewards[n_envs], done[n_envs,], info[n_envs]
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, done = map(np.array, zip(*results))
        
        # reset environments automatically
        for i in range(len(self.envs)):
            if done[i]:
                new_obs[i] = self.envs[i].reset()
        
        return new_obs, rewards, done


# To wrap env into right height and width environment
class PolycraftEnv(gym.Env):
    def __init__(self, height=256, width=256, color=True, crop=lambda img: img, 
                 n_frames=4, reward_scale=0.1,):
        """A Polycraft Gym wrapper for building up an environment"""
        super(PolycraftEnv, self).__init__()

        self.gym = PolycraftGym.Gym('127.0.0.1', 9000)
        self.gym.sock_connect()

        self.img = None
        self.img_size = (height, width)
        self.crop=crop
        self.color=color

        self.reward_scale = reward_scale
        self.time_step = 0
        self.curr_score = 0.
        
        # Define observation space
        n_channels = (3 * n_frames) if color else n_frames
        obs_shape = [height,width,n_channels]
        self.observation_space = Box(0.0, 255.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')

        # Define action space
        # Action Space: 
        self.action_dic = ['PLACE_MACGUFFIN',
                           'SMOOTH_MOVE W',
                           'SMOOTH_MOVE E',
                           'SMOOTH_MOVE N',
                           'SMOOTH_MOVE S',
                           'SMOOTH_TURN 45',
                           'SMOOTH_TURN -45',
                           'SMOOTH_TILT FORWARD',
                           'SMOOTH_TILT DOWN'
                           'LOOK_EAST',
                           'LOOK_WEST',
                           'LOOK_NORTH',
                           'LOOK_SOUTH',
                           ]
        self.action_space = Discrete(len(self.action_dic))


    
    def reset(self):
        """resets breakout, returns initial frames"""
        # self.gym.send_command('START')
        # self.gym.send_command('RESET domain ../experiments/hgv1_1.json')

        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer()
        self.time_step = 0
        self.curr_score = 0
        return self.framebuffer
    
    def step(self,action):
        """plays breakout for 1 step, returns frame buffer"""
        data = self.gym.step_command(self.action_dic[action])
        print(data['command_result'])

        self.update_buffer()
        self.time_step += 1

        # Done
        if data['goal']['goalAchieved']:
            done = True
        else:
            done = False

        # Reward handling
        reward = 0
        if data['command_result']['result'] == 'SUCCESS':
            reward += 5
        elif data['command_result']['result'] == 'FAIL':
            reward -= 1
        
        reward = reward - 0.1*data['command_result']['stepCost']
        reward = reward - (self.time_step / 10) * 0.5
        
        #reward = self.curr_score + reward * self.reward_scale
        
        return self.framebuffer, reward, done
    
    def render(self, mode='human'):

        print('Current scores: ', self.curr_score)
        if not self.img:
            data = self.gym.step_command('SENSE_SCREEN')
            self.img = self.preproc_image(data)

        plt.imshow(self.img, interpolation='none')
        plt.show()

    
    ### image processing ###
    
    def update_buffer(self):
        data = self.gym.step_command('SENSE_SCREEN')
        self.img = self.preproc_image(data)
        offset = 3 if self.color else 1
        axis = -1
        cropped_framebuffer = self.framebuffer[:,:,:-offset]
        self.framebuffer = np.concatenate([self.img, cropped_framebuffer], axis = axis)

    def preproc_image(self, data):
        """what happens to the observation"""
        img_array = np.array(data['screen']['img'], dtype=np.uint32)
        img_array = img_array.view(np.uint8).view(np.uint8).reshape(img_array.shape+(4,))[..., :3]
        img_array = np.reshape(img_array, self.img_size+(3,))
        # img_array = np.flip(img_array, 0)
        # img = np.flip(img_array, 2)

        img = self.crop(img_array)
        if not self.color:
            img = img.mean(-1, keepdims=True)
        img = img.astype('float32') / 255.
        return img