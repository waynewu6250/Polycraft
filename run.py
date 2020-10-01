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
import json
from config import Config

def randomize(domain_file):

    domain_file = domain_file[3:]
    with open(domain_file, 'r') as f:
        setting = json.load(f)
    
    num = [int(i) for i in np.random.randint(10, 30, 6)]
    angle = int(np.random.randint(0, 360, 1)[0])
    setting['features'][0]['pos'] = [num[0], 4, num[1]]
    setting['features'][0]['lookDir'] = [0, angle, 0]
    setting['features'][2]['pos'] = [num[2], 4, num[3]]
    setting['features'][4]['blockList'][0]['blockPos'] = [num[4], 4, num[5]]
    with open('../polycraft_game/experiments/hgv1_1.json', 'w') as f:
        f.write(json.dumps(setting))

def run(opt):

    # randomize(opt.domain_file)

    env = PolycraftHGEnv(opt=opt)
    env = wrap_func(env)

    scores = []
    eps = 0.
    state = env.reset(opt.domain_file)
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
                state = env.reset(opt.domain_file)
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
    