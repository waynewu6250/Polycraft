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
from config import Config

def run(opt):

    env = PolycraftHGEnv(domain_file=opt.domain_file, opt=opt)
    env = wrap_func(env)

    scores = []
    eps = 0.
    state = env.reset()
    counter = 0                                 
    
    for i_episode in range(5):
        
        print()

        score = 0                                          # initialize the score
        agent = Agent(state_size = opt.state_size, action_size = opt.action_Size, seed = 0, opt=opt)
        
        while True:
            action = agent.select_act(state,eps)           # select an action
            next_state, reward, done, info = env.step(action)   # get the next state
            loss = agent.step(state,action,reward,next_state,done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            counter += 1
            if done or counter == 250:                     # exit loop if episode finished
                state = env.reset()
                break
        scores.append(score)
        #print("Score: {}".format(score))
    print('Avg score:',np.mean(scores))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='all', dest='mode')
    args = parser.parse_args()

    opt = Config(args)
    run(opt)
    