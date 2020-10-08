import socket, json, time
import numpy as np
from matplotlib import pyplot as plt

# connect to socket'
HOST = '127.0.0.1'
PORT = 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT)) 


def MC(command):
    "function that enable the communication with minecraft"
    print( command )
    sock.send(str.encode(command + '\n'))
    if not command.startswith('START'):
      BUFF_SIZE = 4096  # 4 KiB
      data = b''
      while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        time.sleep(.05)
        if len(part) < BUFF_SIZE:
          # either 0 or end of data
          break
      data_dict = json.loads(data)
      return data_dict




## start game world
MC('START')

## start Hunter_Gatherer game
MC('RESET domain ../experiments/hgv1_1.json')



############ APIs for AI-bot to get info from game ######################
## get player locatin
data = MC('SENSE_LOCATIONS')
data['player']['pos']
data['player']['yaw']
data['player']['pitch']

## get screen information
data = MC('SENSE_SCREEN')
img_array = np.array(data['screen']['img'], dtype=np.uint32)
img_array = img_array.view(np.uint8).view(np.uint8).reshape(img_array.shape+(4,))[..., :3]
img_array = np.reshape(img_array, (256, 256, 3))
img_array = np.flip(img_array, 0)
img_array = np.flip(img_array, 2)

print(img_array)
plt.imshow(img_array, interpolation='none')
plt.show()


############ APIs for AI-bot to send actions commands into the game ######################
MC('SMOOTH_TILT FORWARD')               # look horizontally
MC('LOOK_EAST')                         # face east
MC('SMOOTH_MOVE W')                     # move forward one step
MC('SMOOTH_TURN 90')                    # turn 90 degrees, value can be multiples of 15
MC('SMOOTH_TURN -90')


