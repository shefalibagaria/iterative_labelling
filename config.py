import json


class Config():
    """Config class
    """
    def __init__(self, tag):
        self.tag = tag
        self.path = f'runs/{self.tag}'
        self.data_path = ''
        self.net_type = 'cnn'
        self.l = 64     # what is l
        self.n_channels = 1
        self.n_phases = 3
        # Training hyperparams
        self.batch_size = 2
        self.beta1 = 0.9       # what are beta 1 and 2 
        self.beta2 = 0.999
        self.num_epochs = 10000
        self.iters = 1
        self.lrg = 0.0001       # what is lrg
        self.lr = 0.001
        self.Lambda = 10        # what is lambda
        self.critic_iters = 10  # what is critic_iters
        self.lz = 4             # what is lz
        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'
        self.nz = 100           # what is nz
        # Architecture
        self.lays = 4
        self.laysd = 5          # what is laysd
        # kernel sizes
        # self.dk, self.gk = [4]*self.laysd, [3]*self.lays
        # self.ds, self.gs = [2]*self.laysd, [1]*self.lays
        # self.df, self.gf = [self.n_phases, 64, 128, 256, 512, 1], [
        #     16, 16, 16, 16, self.n_phases]
        # self.dp = [1, 1, 1, 1, 0]
        # self.gp = [1, 1, 1, 1, 1]

        self.k = [3]*self.lays
        self.s = [1]*self.lays
        self.p = [1]*self.lays
        self.f = [self.n_channels, 16, 16, 16, self.n_phases]


    def save(self):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.path}/config.json', 'w') as f:
            json.dump(j, f)

    def load(self):
        with open(f'{self.path}/config.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)

    def get_net_params(self):
        return self.k, self.s, self.f, self.p
    
    def get_train_params(self):
        return self.l, self.batch_size, self.beta1, self.beta2, self.num_epochs, self. s, self.lrg, self.lr, self.Lambda, self.critic_iters, self.lz, self.nz


