# PolycraftEnv.py
#
# Defines OpenAI Gym environments for Polycraft.
#
# Georgia Tech

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
    
    def __init__(self, opt):
        self.opt = opt
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect((HOST, PORT))
        self.process_command('start')
        time.sleep(1)
        
        if self.opt.mode == 'all':
            self.action_names = [
                'move w',   # forward
                # 'move a',   # left
                # 'move d',   # right
                # 'move x',   # back
                'turn -45', # turn left
                'turn 45',  # turn right
                'compound'] #place_macguffin
                # 'SMOOTH_TILT FORWARD']
                #]
        else:
            self.action_names = [
                'move w',   # forward
                'move x',   # backward
                'turn -45', # turn left
                'turn 45']  # turn right
    
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

        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

        self.prev_screen = None
        self.get_macguffin = False
    
    # ----- Required methods of environment

    def reset(self, domain_file):
        command = "reset domain " + domain_file
        state_dict = self.process_command(command)
        # Here we don't have valid observation
        # observation = self.generate_observation(state_dict)
        time.sleep(5)
        screen = self.process_command('SENSE_SCREEN')
        if 'screen' not in screen and self.prev_screen:
            screen = self.prev_screen
        else:
            self.prev_screen = screen
        screen = screen['screen']['img']
        return screen
        
    def step(self, action):
        # action assumed to be an integer index into action_names list
        act_dict, state_dict, screen = self.process_action(action)
        if 'screen' not in screen and self.prev_screen:
            screen = self.prev_screen
        else:
            self.prev_screen = screen
        screen = screen['screen']['img']
        reward = self.get_reward(state_dict, act_dict)
        done = self.goal_achieved(state_dict)
        info = {}
        
        # print("\nACTION: " + self.action_names[action])
        # observation = self.generate_observation(state_dict)
        # self.print_observation(observation)
        # print("REWARD = %.2f" % reward)
        return screen, reward, done, info
    
    def render(self, mode='human', close=False):
        # polycraft rendering handled externally
        return None
        
    def close(self):
        self.sock.close()
        return None

    # ----- Utility methods of environment
    
    def process_action(self, action_index):
        action = self.action_names[action_index]

        if self.opt.mode == 'all':
            if action == 'compound':
                act_dict = self.process_command('place_macguffin')
                act_dict = self.process_command('SMOOTH_TILT FORWARD')
            else:
                act_dict = self.process_command(action)
        else:
            act_dict = self.process_command(action)
        
        time.sleep(0.001)
        dict = self.process_command('sense_all')
        screen = self.process_command('SENSE_SCREEN')
        #time.sleep(0.25)
        return act_dict, dict, screen

    def process_command(self, command):
        self.sock.send(str.encode(command + '\n'))
        BUFF_SIZE = 4096  # 4 KiB
        data = b''
        while True:
            try:
                part = self.sock.recv(BUFF_SIZE)
            except:
                part = b''
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
        if self.opt.mode == 'all':
            goal_ach = False
            if 'macGuffinPos' in state_dict:
                goal = state_dict['macGuffinPos']
                mac_x = goal[0]
                mac_y = goal[2]
                mac_z = goal[1]
                goal_ach = (mac_x == 10 and mac_z == 4 and mac_y == 12)
        else:
            goal_ach = self.have_macguffin(state_dict)
        return goal_ach

    def get_reward(self, state_dict, act_dict):

        if self.opt.mode == 'all':
            reward = -0.1
            if act_dict['command_result']['command'] == 'turn':
                reward -= 0.1
            elif act_dict['command_result']['command'] == 'SMOOTH_TILT':
                reward -= 0.3
            if act_dict['command_result']['result'] == 'FAIL':
                reward -= 0.5
            if self.have_macguffin(state_dict):
                if not self.get_macguffin:
                    self.get_macguffin = True
                    reward += 100.0
                else:
                    reward += 0.1
            else:
                if self.get_macguffin:
                    reward -= 150.0
                    self.get_macguffin = False
            if self.goal_achieved(state_dict):
                reward += 1000.0
        
        else:
            # move: -0.3
            # turn: -0.5
            # fail: -1.0
            # goal: 100
            reward = -0.1 #0.0
            if act_dict['command_result']['result'] == 'SUCCESS':
                reward -= 0.2
            else:
                reward -= 0.9
            if act_dict['command_result']['command'] == 'turn':
                reward -= 0.2
            # if act_dict['command_result']['command'] == 'turn':
            #     reward -= 0.2
            # if act_dict['command_result']['result'] == 'SUCCESS':
            #     reward += 0.3
            # else:
            #     reward -= 0.2
            if self.have_macguffin(state_dict):
                reward = 50.0
            if self.goal_achieved(state_dict):
                reward = 200.0 #100
        

        return reward

