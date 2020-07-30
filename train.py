from PolycraftEnv import PolycraftHGEnv
from wrapper import wrap_func
import numpy as np
from model import QNetwork
from buffer import Replay_Buffer
from Agent import Agent
from collections import deque
from tqdm import tqdm
import torch
import os

Batch_Size = 1
Action_Size = 3

env = PolycraftHGEnv(domain_file='../experiments/hgv1_1.json')
env = wrap_func(env)

agent = Agent(state_size = 8, action_size = Action_Size, seed = 0)
if os.path.exists('saved_model.pth'):
    print('Load pretrained model...')
    agent.qnetwork_local.load_state_dict(torch.load('saved_model.pth'))

def dqn_unity(num_episodes = 2000,  eps_start = 1, eps_decay=0.995, eps_end = 0.01):
    
    scores = [] # list of scores from each episode
    score_window = deque(maxlen = 100) # a deque of 100 episode scores to average
    eps = eps_start
    state = env.reset()
    
    for i_episode in tqdm(range(1,num_episodes+1)):
        
        score = 0
        counter = 0
        while True:
                   
            action = agent.select_act(state,eps)           # select an action
            next_state, reward, done, info = env.step(action)
            agent.step(state,action,reward,next_state,done)
            score += reward
            state = next_state
            counter += 1
            # print('==========================')
      
            # print('Current Step: ', counter)
            # print('==========================')
            
            if done or counter == 100:
                state = env.reset()
                break
        scores.append(score)
        score_window.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'saved_model.pth')
        if np.mean(score_window)>=99.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'saved_model.pth')
            
            
    return scores


scores = dqn_unity()
