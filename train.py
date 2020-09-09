import numpy as np
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
import torch
import os

from PolycraftEnv import PolycraftHGEnv
from wrapper import wrap_func
from model import QNetwork
from Agent import Agent
from config import Config

def dqn_unity(opt):

    # environment
    env = PolycraftHGEnv(domain_file=opt.domain_file, opt=opt)
    env = wrap_func(env)

    # agent
    agent = Agent(state_size = opt.state_size, action_size = opt.action_Size, seed = 0, opt=opt)
    
    # parameters
    scores = [] # list of scores from each episode
    losses = [] # list of losses
    score_window = deque(maxlen = 100) # a deque of 100 episode scores to average
    eps = opt.eps_start
    state = env.reset()
    counter = 0

    # create figure
    plt.figure(figsize=(6,3), dpi=80)
    plt.ion()
    ewma = lambda x, span=100: DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values
    
    # start training
    for i_episode in tqdm(range(1,opt.num_episodes+1)):

        print()
        
        score = 0
        aloss = 0
        while True:
            
            action = agent.select_act(state,eps)           # select an action
            next_state, reward, done, info = env.step(action)
            loss = agent.step(state,action,reward,next_state,done)
            if loss:
                print('Current Loss: {:.4f}'.format(loss.item()))
                aloss += loss.item()
            score += reward
            state = next_state
            counter += 1
            # print('==========================')
      
            # print('Current Step: ', counter)
            # print('==========================')
            
            if done or counter % 250 == 0:
                state = env.reset()
                break
        scores.append(score)
        score_window.append(score)
        if aloss != 0:
            losses.append(aloss)
        eps = max(opt.eps_end, opt.eps_decay*eps) # decrease epsilon
        
        if i_episode % 1 == 0:
            print('\rAverage Score: {:.2f}'.format(np.mean(score_window)))
            # sanity check
            for param in agent.qnetwork_local.parameters():
                print(param[0][0][0])
                break
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='all', dest='mode')
    args = parser.parse_args()

    opt = Config(args)
    scores = dqn_unity(opt)
