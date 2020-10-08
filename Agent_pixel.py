import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple,deque

from model_pixel import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_input_chnl, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            num_input_chnl (int): number of input channels
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.num_input_chnl = num_input_chnl
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        self.opt = opt

        # Q-Network
        self.qnetwork_local = QNetwork(num_input_chnl, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(num_input_chnl, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.opt.LR, weight_decay=self.opt.REGULARIZATION)

        # Load model
        if os.path.exists(opt.model_path):
            print('Load pretrained model...')
            self.qnetwork_local.load_state_dict(torch.load(opt.local_model_path,        map_location=lambda storage, loc: storage))
            self.qnetwork_target.load_state_dict(torch.load(opt.target_model_path, map_location=lambda storage, loc: storage))
            # sanity check
            for param in self.qnetwork_local.parameters():
                print(param[0][0][0])
                break
        else:
            print('Train from scratch...')

        # Replay memory
        self.memory = ReplayBuffer(action_size, seed, opt)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def select_act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state) #same as self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def step(self, state, action, reward, next_state, done, is_training=True):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.opt.UPDATE_EVERY
        
        d_error = None
        if self.t_step == 0:
            # If enough samples are available in memory and in training mode, then get random subset and learn
            if len(self.memory) > self.opt.batch_size and is_training == True:
                experiences = self.memory.sample_augmented_experience()
                d_error = self.learn(experiences, self.opt.gamma)
            
        return d_error
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## compute and minimize the loss
        qs_local = self.qnetwork_local.forward(states)
        qsa_local = qs_local[torch.arange(self.opt.batch_size, dtype=torch.long), actions.reshape(self.opt.batch_size)]
        qsa_local = qsa_local.reshape((self.opt.batch_size,1))

        # DQN Target
        qs_target = self.qnetwork_target.forward(next_states)
        qsa_target, _ = torch.max(qs_target, dim=1) #using the greedy policy (q-learning)
        qsa_target = qsa_target * (1 - dones.reshape(self.opt.batch_size)) #target qsa value is zero when episode is complete
        qsa_target = qsa_target.reshape((self.opt.batch_size,1))
        TD_target = rewards + gamma * qsa_target
        
        loss = F.mse_loss(qsa_local, TD_target) #much faster than the above loss function
        #print(loss)
        #minimize the loss
        self.optimizer.zero_grad() #clears the gradients
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.TAU)

        return loss
    
    
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

    def augment_state(self, state):
        # Augment the state to include previous observations and actions
        input_image_shape = self.memory.input_image_shape
        if len(self.memory) >= 2:
            prev_idx = len(self.memory)-1
            prev_prev_idx = prev_idx-1
            prev_e = self.memory.memory[prev_idx]
            prev_prev_e = self.memory.memory[prev_prev_idx]

            #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
            prev_e_a = np.ones((1,1,input_image_shape[0],input_image_shape[1]))*prev_e.action
            prev_prev_e_a = np.ones((1,1,input_image_shape[0],input_image_shape[1]))*prev_prev_e.action
            aug_state = np.concatenate((prev_prev_e.state, prev_prev_e_a, prev_e.state, prev_e_a, state), axis=1)
        else:
            #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
            initial_action = 0
            prev_e_a = np.ones((1,1,input_image_shape[0],input_image_shape[1]))*initial_action
            prev_prev_e_a = np.ones((1,1,input_image_shape[0],input_image_shape[1]))*initial_action
            aug_state = np.concatenate((state, prev_prev_e_a, state, prev_e_a, state), axis=1)

        return aug_state
    


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        ###self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) #doesn't like tuple to be defined inside class when using pickle
        self.experience = Experience
        self.seed = seed
        random.seed(seed) #returns None
        self.input_image_shape = (84,84)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample_old(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def sample_augmented_experience(self):
        """Randomly sample a batch of experiences from memory."""
        #Note: the experiences are store in the memory in chronoogical order

        #experiences = list(self.memory)[0:self.batch_size] #get experiences in order

        aug_states = [] #augment state
        actions = []
        rewards = []
        aug_next_states = [] #augment next state
        dones = []
        while len(aug_states) < self.batch_size:
            idx = random.sample(range(len(self.memory)), k=1)[0]
            #idx = 3+len(aug_states) #take experiences in order and in agent.step make sure 'len(self.memory) > BATCH_SIZE+5'
            e = self.memory[idx]
            if e is None or (idx-2) < 0 or (idx+1) >= len(self.memory):
                continue
            else:
                prev_e = self.memory[idx-1]
                prev_prev_e = self.memory[idx-2]
                next_e = self.memory[idx+1]

            #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
            prev_e_a = np.ones((1,1,self.input_image_shape[0],self.input_image_shape[1]))*prev_e.action
            prev_prev_e_a = np.ones((1,1,self.input_image_shape[0],self.input_image_shape[1]))*prev_prev_e.action
            aug_state = np.concatenate((prev_prev_e.state, prev_prev_e_a, prev_e.state, prev_e_a, e.state), axis=1)
            aug_states.append(aug_state)
            actions.append(e.action)
            rewards.append(e.reward)
            e_a = np.ones((1,1,self.input_image_shape[0],self.input_image_shape[1]))*e.action
            aug_next_state = np.concatenate((prev_e.state, prev_e_a, e.state, e_a, next_e.state), axis=1)
            aug_next_states.append(aug_next_state)
            dones.append(e.done)

        #augment state is of shape Nx11x84x84
        states = torch.from_numpy(np.vstack([s for s in aug_states])).float().to(device)
        actions = torch.from_numpy(np.vstack([a for a in actions])).long().to(device)
        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float().to(device)
        next_states = torch.from_numpy(np.vstack([ns for ns in aug_next_states])).float().to(device)
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)