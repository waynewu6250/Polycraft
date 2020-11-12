import numpy as np
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
import torch
import json
import os
import pickle

from PolycraftEnv import PolycraftHGEnv
from wrapper import wrap_func
from config import Config

# from model import QNetwork
# from Agent import Agent
from model_pixel import QNetwork
from Agent_pixel import Agent

def randomize(domain_file):

    domain_file = domain_file[3:]
    with open(domain_file, 'r') as f:
        setting = json.load(f)
    
    location = [7, 10, 20, 23] #[1, 3, 4, 6, 7, 10, 20, 26, 30]
    num = [int(i) for i in np.random.choice(location, 6)]
    angle = int(np.random.choice([0, 90, 180, 270], 1))
    setting['features'][0]['pos'] = [num[0], 4, num[1]]
    setting['features'][0]['lookDir'] = [0, angle, 0]
    setting['features'][2]['pos'] = [num[2], 4, num[3]]
    setting['features'][4]['blockList'][0]['blockPos'] = [num[4], 4, num[5]]
    with open('../polycraft_game/experiments/hgv1_1.json', 'w') as f:
        f.write(json.dumps(setting))

def run_training(args):

    # randomize(opt.domain_file)
    opt = Config(args)
    env = PolycraftHGEnv(opt=opt)
    env = wrap_func(env)

    scores = []
    eps = 0.1
    state = env.reset(opt.domain_file)      
    dones = 0
    success_turning = torch.load('success_turning_all.pth') if os.path.exists('success_turning_all.pth') else []      
    opt.local_model_path = 'checkpoints/saved_model_local_second.pth'
    opt.target_model_path = 'checkpoints/saved_model_target_second.pth'
    second_agent = Agent(num_input_chnl = opt.num_input_chnl, action_size = opt.action_Size, seed = 0, opt=opt)

    for i_episode in tqdm(range(1,opt.num_episodes+1)):

        opt = Config(args)

        if i_episode % 300 == 0 or dones == 100:
            if dones == 100:
                success_turning.append(i_episode)
            randomize(opt.domain_file)
            dones = 0
        
        print()
        first_agent = Agent(num_input_chnl = opt.num_input_chnl, action_size = opt.action_Size, seed = 0, opt=opt)
        
        score = 0                                          # initialize the score
        counter = 0
        while True:
            # action = agent.select_act(state,eps)           # select an action
            aug_state = first_agent.augment_state(state)
            action = first_agent.select_act(aug_state,eps)           # select an action
            next_state, reward, done, info = env.step(action)   # get the next state
            loss = first_agent.step(state,action,reward,next_state,done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            counter += 1
            if counter == 250:                     # exit loop if episode finished
                state = env.reset(opt.domain_file)
                break
            elif done:
                opt.local_model_path = 'checkpoints/saved_model_local_second.pth'
                opt.target_model_path = 'checkpoints/saved_model_target_second.pth'
                scores, do = dqn_unity(opt, second_agent)
                dones += int(do)

def dqn_unity(opt, agent):
    
    # parameters
    scores = [] # list of scores from each episode
    losses = [] # list of losses
    score_window = deque(maxlen = 100) # a deque of 100 episode scores to average
    eps = opt.eps_start
    state = env.reset(opt.domain_file)

    # create figure
    plt.figure(figsize=(6,3), dpi=80)
    plt.ion()
    ewma = lambda x, span=100: DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values
    
    # start training
    score = 0
    aloss = 0
    counter = 0
    while True:
        aug_state = agent.augment_state(state)
        action = agent.select_act(aug_state,eps)           # select an action
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
        
        if done or counter == 250:
            state = env.reset(opt.domain_file)
            break
    scores.append(score)
    score_window.append(score)
    if aloss != 0:
        losses.append(aloss)
    eps = max(opt.eps_end, opt.eps_decay*eps) # decrease epsilon
    
    if i_episode % 20 == 0:
        print('\rAverage Score: {:.2f}'.format(np.mean(score_window)))
        # sanity check
        for param in agent.qnetwork_local.parameters():
            print(param[0][0][0])
            break
        torch.save(agent.qnetwork_local.state_dict(), opt.local_model_path)
        torch.save(agent.qnetwork_target.state_dict(), opt.target_model_path)
        torch.save(success_turning, 'success_turning.pth')
        with open(opt.buffer_path, "wb") as mf:
            mem_to_save = deque(list(agent.memory.memory)[-50000:], maxlen=100000) #agent.memory.memory
            pickle.dump(mem_to_save, mf, pickle.HIGHEST_PROTOCOL)
    if np.mean(score_window)>=99.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)))
    
    # Create figure
    plt.cla()
    plt.subplot(1,2,1)
    if scores != []:
        plt.cla()
        plt.scatter(range(len(scores)), scores, label='rewards', s=5)
        #plt.plot(ewma(np.array(scores),span=10), marker='.', label='rewards ewma@1000')
        plt.title("Session rewards"); plt.grid(); plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(losses, label='loss')
    plt.plot(ewma(np.array(losses),span=1000), label='loss ewma@1000')
    plt.title("Training Losses"); plt.grid(); plt.legend()

    plt.pause(0.005)
    plt.savefig('results/score.png')
    print(success_turning)
    print(eps)
    
    plt.ioff()
            
    return scores, done


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='all', dest='mode')
    args = parser.parse_args()

    scores = run_training(args)
