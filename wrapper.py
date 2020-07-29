import numpy as np
from collections import deque
import gym
from gym import spaces
from PIL import Image
import matplotlib.pyplot as plt

from PolycraftEnv import PolycraftHGEnv

# 實作每4個frames當作一次sample
class MaxAndSkipEnv:
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        self.env = env
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        # 選倒數兩個frame中較大的那一個，但我不太清楚為何要這樣?
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        # 清掉buffer，並掛上初始obs當作deque的初始狀態
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

# 重新實作經過前處理的step與reset
class ProcessFrame:
    def __init__(self, env=None):
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_frame(obs), reward, done, info

    def reset(self):
        return self._process_frame(self.env.reset())
    
    def _process_frame(self, frame):
        img_array = np.array(frame, dtype=np.uint32)
        img_array = img_array.view(np.uint8).view(np.uint8).reshape(img_array.shape+(4,))[..., :3]
        img_array = np.reshape(img_array, (256, 256, 3))
        img_array = np.flip(img_array, 0)
        img = np.flip(img_array, 2)
        # plt.imshow(img_array, interpolation='none')
        # plt.show()
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = Image.fromarray(img)
        resized_screen = img.resize((84, 110), Image.BILINEAR)
        resized_screen = np.array(resized_screen)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        # plt.imshow(x_t.squeeze(2))
        # plt.show()
        return x_t.astype(np.uint8)

def wrap_func(env):
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame(env)
    return env
