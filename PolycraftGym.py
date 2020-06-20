import os, subprocess, socket, json
from pathlib import Path
import time

class Gym(object):

    host = '127.0.0.1'
    port = 9000

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start_client(self, scene):
        # this function is currently broken!
        path = Path(os.getcwd())
        # print(str(path.parent) + '\\gradlew RunClient')
        # os.system('\"' + str(path.parent) + '\\gradlew RunClient' + '\"')
        # subprocess.run([str(path.parent) + '\\gradlew', 'setupDecompWorkspace'])

    def setup_scene(self, scene):
        self.send_command('RESET domain ' + scene)

    def sock_connect(self):
        self.sock.connect((self.host, self.port))

    def sock_close(self):
        self.sock.close()

    def send_command(self, command):
        self.sock.send(str.encode(command + '\n'))

    def step_command(self, command):
        self.sock.send(str.encode(command + '\n'))
        data = self.recvall(command)
        data_dict = json.loads(data)
        return data_dict

    def action_sample_LL(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def step_command_LL(self, action_space, command):
        self.sock.send(str.encode(command + '\n'))
        data = self.recvall(command)
        data_dict = json.loads(data)
        return data_dict

    def recvall(self, command):
        BUFF_SIZE = 4096  # 4 KiB
        data = b''
        while True:
            part = self.sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                # either 0 or end of data
                break
        return data
