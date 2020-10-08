class Config:

    def __init__(self, args):

        self.mode = args.mode

        if self.mode == 'all':
            self.domain_file = '../experiments/hgv1_0.json'
            self.action_Size = 4
        
        elif self.mode == 'single':
            self.domain_file = '../experiments/hgv1_1.json'
            self.action_Size = 3

        # environment
        self.state_size = 8
        self.num_episodes = 3000
        self.eps_start = 0.3 # 1
        self.eps_decay=0.995
        self.eps_end = 0.1 #0.01

        # agent
        self.buffer_size = int(1e5)  # replay buffer size
        self.batch_size = 32         # minibatch size
        self.gamma = 0.99            # discount factor
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR = 5e-4               # learning rate 
        self.UPDATE_EVERY = 4        # how often to update the network

        # path
        prefix = ''
        #prefix = '_BEST'
        self.model_path = 'saved_model{}.pth'.format(prefix)
        self.frame_path = 'saved_frames{}.pkl'.format('_BEST') #'_BEST'

        self.is_recover = True       # whether to recover old buffer


