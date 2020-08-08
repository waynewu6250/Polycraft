from PolycraftEnv import PolycraftHGEnv
from wrapper import wrap_func
import numpy as np
from model import QNetwork
from Agent import Agent
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
import torch
import os

Batch_Size = 32
Action_Size = 3

env = PolycraftHGEnv(domain_file='../experiments/hgv1_1.json')
env = wrap_func(env)

agent = Agent(state_size = 8, action_size = Action_Size, seed = 0)
if os.path.exists('saved_model.pth'):
    print('Load pretrained model...')
    agent.qnetwork_local.load_state_dict(torch.load('saved_model.pth'))
else:
    print('Train from scratch...')

def dqn_unity(num_episodes = 3000,  eps_start = 1, eps_decay=0.995, eps_end = 0.01):
    
    scores = [] # list of scores from each episode
    losses = [] # list of losses
    score_window = deque(maxlen = 100) # a deque of 100 episode scores to average
    eps = eps_start
    state = env.reset()
    counter = 0

    # Create Figure
    plt.figure(figsize=(6,3), dpi=80)
    plt.ion()
    ewma = lambda x, span=100: DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values
    
    for i_episode in tqdm(range(1,num_episodes+1)):

        print()
        
        score = 0
        aloss = 0
        while True:
            
            if counter <= Batch_Size:
                is_random = True
            else:
                is_random = False
            
            action = agent.select_act(state,eps,is_random)           # select an action
            next_state, reward, done, info = env.step(action)
            loss = agent.step(state,action,reward,next_state,done,is_random)
            if loss:
                print('Current Loss: {:.4f}'.format(loss.item()))
                aloss += loss.item()
            score += reward
            state = next_state
            counter += 1
            # print('==========================')
      
            # print('Current Step: ', counter)
            # print('==========================')
            
            if done or counter % 100 == 0:
                state = env.reset()
                break
        scores.append(score)
        score_window.append(score)
        if aloss != 0:
            losses.append(aloss)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        if i_episode % 1 == 0:
            print('\rAverage Score: {:.2f}'.format(np.mean(score_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'saved_model.pth')
        if np.mean(score_window)>=99.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'saved_model.pth')
        
        # Create figure
        plt.cla()
        plt.subplot(1,2,1)
        if scores != []:
            plt.cla()
            plt.plot(scores, label='rewards')
            plt.plot(ewma(np.array(scores),span=10), marker='.', label='rewards ewma@1000')
            plt.title("Session rewards"); plt.grid(); plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(losses, label='loss')
        plt.plot(ewma(np.array(losses),span=1000), label='loss ewma@1000')
        plt.title("Training Losses"); plt.grid(); plt.legend()

        plt.pause(0.005)
        plt.savefig('score.png')
    
    plt.ioff()
            
            
    return scores


scores = dqn_unity()
