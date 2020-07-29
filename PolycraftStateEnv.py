# PolycraftEnv.py
#
# Defines OpenAI Gym environments for Polycraft.
#
# Washington State University

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import json
import socket
import numpy as np
from gym import spaces

HOST = "127.0.0.1"
PORT = 9000

# Hunter-Gatherer environment

class PolycraftHGEnv:
    
    def __init__(self, domain_file):
        self.domain_file = domain_file
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, PORT))
        self.process_command('start')
        time.sleep(1)
        
        self.action_names = [
            'move w',   # forward
            'move a',   # left
            'move d',   # right
            'move x',   # back
            'turn -15', # turn left
            'turn 15']  # turn right
            #'place_macguffin'
            #]
    
        self.facing_names = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    
        self.state_names = [
            'player_pos_x',
            'player_pos_y',
            'player_pos_z',
            'player_facing',
            'dest_pos_x',
            'dest_pos_y',
            'dest_pos_z',
            'have_macguffin'
            ]

        obs_low = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        obs_high = np.array([100, 100, 100, 3, 100, 100, 100, 1])
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.uint32)
    
    # ----- Required methods of environment
        
    def step(self, action):
        # action assumed to be an integer index into action_names list
        state_dict = self.process_action(action)
        observation = self.generate_observation(state_dict)
        reward = self.get_reward(state_dict)
        done = self.goal_achieved(state_dict)
        info = {}
        print("\nACTION: " + self.action_names[action])
        self.print_observation(observation)
        print("REWARD = %.2f" % reward)
        return observation, reward, done, info
    
    def reset(self):
        command = "reset domain " + self.domain_file
        self.previous_score = 0.0
        state_dict = self.process_command(command)
        observation = self.generate_observation(state_dict)
        time.sleep(2)
        return observation
    
    def render(self, mode='human', close=False):
        # polycraft rendering handled externally
        return None
        
    def close(self):
        self.sock.close()
        return None

    # ----- Utility methods of environment
    
    def process_action(self, action_index):
        action = self.action_names[action_index]
        dict = self.process_command(action)
        #time.sleep(0.25)
        dict = self.process_command('sense_all')
        #time.sleep(0.25)
        return dict

    def process_command(self, command):
        self.sock.send(str.encode(command + '\n'))
        BUFF_SIZE = 4096  # 4 KiB
        data = b''
        while True:
            part = self.sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                 # either 0 or end of data
                break
        data_dict = json.loads(data)
        return data_dict

    def generate_observation(self, state_dict):
        # Player position
        player_pos_x = player_pos_y = player_pos_z = player_facing = 0
        if 'player' in state_dict:
            player = state_dict['player']
            player_pos = player['pos']
            player_pos_x = player_pos[0]
            player_pos_y = player_pos[2]
            player_pos_z = player_pos[1]
            player_facing = self.facing_names.index(player['facing'])
        # Destination position
        dest_pos_x = dest_pos_y = dest_pos_z = 0
        if 'destinationPos' in state_dict:
            dest_pos = state_dict['destinationPos']
            dest_pos_x = dest_pos[0]
            dest_pos_y = dest_pos[2]
            dest_pos_z = dest_pos[1]
        # Player have macguffin?
        have_macguffin = 0
        if self.have_macguffin(state_dict):
            have_macguffin = 1
        return np.array([player_pos_x, player_pos_y, player_pos_z, player_facing,
                dest_pos_x, dest_pos_y, dest_pos_z, have_macguffin])
    
    def print_observation(self, obs):
        playerStr = "player: " + str([obs[0], obs[1], obs[2]]) + " facing " + self.facing_names[int(obs[3])]
        destStr = "dest: " + str([obs[4], obs[5], obs[6]])
        if obs[7] == 0:
            macguffinStr = "macguffin: no"
        else:
            macguffinStr = "macguffin: yes"
        print("STATE: " + playerStr + ", " + destStr + ", " + macguffinStr)

    def have_macguffin(self, state_dict):
        
        have_macguffin = False
        if 'inventory' in state_dict:
            inventory = state_dict['inventory']
            if '0' in inventory:
                selectedItem = inventory['0']
                item = selectedItem['item']
                if item == 'polycraft:macguffin':
                    have_macguffin = True
        return have_macguffin

    def goal_achieved(self, state_dict):
        goal_ach = self.have_macguffin(state_dict)
        # goal_ach = False
        # if 'goal' in state_dict:
        #     goal = state_dict['goal']
        #     goal_ach = goal['goalAchieved']
        return goal_ach

    def get_reward(self, state_dict):
        reward = -0.01
        if self.have_macguffin(state_dict):
            reward = 50.0
        if self.goal_achieved(state_dict):
            reward = 100.0
        return reward

