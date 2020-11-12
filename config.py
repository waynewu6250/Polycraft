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
        self.num_input_chnl = 11
        self.num_episodes = 3000
        self.eps_start = 0.1
        self.eps_decay=0.995
        self.eps_end = 0.1

        # agent
        self.buffer_size = int(1e5)  # replay buffer size
        self.batch_size = 32         # minibatch size
        self.gamma = 0.99            # discount factor
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR = 5e-4               # learning rate 
        self.UPDATE_EVERY = 4        # how often to update the network
        self.REGULARIZATION = 1e-4   # regularization parameter

        # path
        prefix = '_5'
        #prefix = '_BEST'
        self.local_model_path = 'checkpoints/saved_model_local{}.pth'.format(prefix)
        self.target_model_path = 'checkpoints/saved_model_target{}.pth'.format(prefix)
        self.buffer_path = 'checkpoints/saved_buffer{}.pkl'.format('') #'_BEST'
        self.frame_path = 'checkpoints/saved_frames{}.pkl'.format('_BEST') #'_BEST'

        self.is_recover = True       # whether to recover old buffer


