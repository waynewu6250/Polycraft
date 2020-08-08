# run this file to see how trained models work.

from PolycraftEnv import PolycraftHGEnv
from wrapper import wrap_func
from model import QNetwork
from Agent import Agent
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from collections import namedtuple,deque

Batch_Size = 32
Action_Size = 3

env = PolycraftHGEnv(domain_file='../experiments/hgv1_1.json')
env = wrap_func(env)

agent = Agent(state_size = 8, action_size = Action_Size, seed = 0)

agent.qnetwork_local.load_state_dict(torch.load('saved_model_BEST2.pth'))

eps = 0.
scores = []
for i in range(5):

    state = env.reset()                                # reset the environment
    
    score = 0                                          # initialize the score
    counter = 0
    while True:
        action = agent.select_act(state,eps,False)     # select an action
        next_state, reward, done, info = env.step(action)   # get the next state
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        counter += 1
        if done or counter == 250:                     # exit loop if episode finished
            break
    scores.append(score)
    #print("Score: {}".format(score))
print('Avg score:',np.mean(scores))
