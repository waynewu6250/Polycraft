import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple,deque

from screen_buffer import ScreenReplayBuffer
from model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    
    def __init__(self, state_size, action_size, seed, opt):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # the state size and action size will be used to generate the Q Network
        self.state_size = state_size
        self.action_size = action_size
        ### random.seed(seed) generates sequence of random numbers by performing some operation on initial value.
        #If same initial value is used, it will generate the same sequence of random numbers
        self.seed = random.seed(seed)
        self.opt = opt

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        if os.path.exists(opt.model_path):
            print('Load pretrained model...')
            self.qnetwork_local.load_state_dict(torch.load(opt.model_path))
            # sanity check
            for param in self.qnetwork_local.parameters():
                print(param[0][0][0])
                break
        else:
            print('Train from scratch...')
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.opt.LR)

        # Replay memory
        self.memory = ScreenReplayBuffer(self.opt.buffer_size, 4, opt)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
      
    def select_act(self, state, eps=0.):
        """selects action based on state and epsilon
        """
        self.last_idx = self.memory.store_frame(state)
        recent_observations = self.memory.encode_recent_observation()
        
        # get the state array from env, convert to tensor
        state = torch.from_numpy(recent_observations / 255.0).float().unsqueeze(0).to(device) 
        # unsqueeze(0) adds a singleton dimension at 0 positon
        # useful because the states are in batches     
        # to(device) moves the tensor to the device memory, cpu or cuda

        if not self.memory.can_sample(self.opt.batch_size):
            # don't update the network
            print('random sampling...')
            return np.random.randint(self.action_size)
        elif not self.opt.is_recover:
            self.memory.store_buffer()
            self.opt.is_recover = True
        
        ## put network in eval mode
        self.qnetwork_local.eval()
        
        #get last_layer of the network to retrive index of the max reward
        with torch.no_grad(): # torch.no_grad() prevents calculating gradients in the following block, so no backward_pass.
            action_values = self.qnetwork_local(state)
        
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(self.action_size)        # select an action
        #random.choice(np.arange(self.action_size)) 
    
    def learn(self,experiences,gamma):
        
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(states / 255.0).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)    
        rewards = torch.from_numpy(rewards).float().to(device)  
        next_states = torch.from_numpy(next_states / 255.0).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)        
        
        # Get max predicted Q values (for next states) from target model
        Q_next_states = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        actions = actions.unsqueeze(1)
        dones = dones.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        
        # detach returns a new tensor detachd from the current graph
        # final layer is (batch_size ,action_size)i.e. (64,4), max(1), will find max in the second dim(1)
        # the new tensor is (64,), we then add a singleton dimensin to it with unsqueeze
        # Q_targets_next is the max reward of the four actons for each of the 64 states
        
        Q_target = rewards + (self.opt.gamma*Q_next_states*(1-dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)
        
        #gather rearranges values in the dimension (1 here) of the input tensor (64,4), 
        #as per the indices in the index tensor provided, actions here...actions carries the index of the next action taken
        # given the state in states. SO only one value will be provided..it coud be either of 0,1,2,3..based on def act and state
        #therefore output is 64,1.with reward corresponding to only that action chosen after the state.
        
        # the rewards generated by q_network local is used for comparison with Q_targets to calc.loss
        #then we update parametrs to min loss
        d_error = torch.sum(torch.abs(Q_expected - Q_target))
        #loss = F.mse_loss(Q_expected,Q_target)
        self.optimizer.zero_grad()
        #loss.backward()
        d_error.backward()
        #Q_expected.backward(d_error.data)
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.TAU)

        return d_error
        
        
    def step(self,state,action,reward,next_state,done):
        
        self.memory.store_effect(self.last_idx, action, reward, done) # 將新的資訊存入buffer中
        if not self.memory.can_sample(self.opt.batch_size):
            return
        
        self.t_step = (self.t_step+1) % self.opt.UPDATE_EVERY # self.t_step will increase by 1 after every step() call
                                                    # that means every time step
        d_error = None
        if self.t_step == 0:
            experiences = self.memory.sample(self.opt.batch_size)
            d_error = self.learn(experiences, self.opt.gamma)
        return d_error
    
    def soft_update(self, local_model, target_model, TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
