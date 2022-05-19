from src.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import tifffile
import time
from PyQt5.QtCore import QObject, pyqtSignal

class TrainWorker(QObject):
    def __init__(self, c, label_mask, Net, overwrite):
        super().__init__()
        self.c = c
        self.label_mask = label_mask
        self.net = Net
        self.overwrite = overwrite
        self.quit_flag = False
        print('init trainworker')
        
    finished = pyqtSignal()
    progress = pyqtSignal(int, float)

    def stop(self):
        self.quit_flag = True

    def train(self):
        """[summary]

        :param c: [description]
        :type c: [type]
        :param Gen: [description]
        :type Gen: [type]
        :param Disc: [description]
        :type Disc: [type]
        :param offline: [description], defaults to True
        :type offline: bool, optional
        """
        c = self.c
        Net = self.net
        datapath = self.c.data_path
        label_mask = self.label_mask
        offline = False

        # Assign torch device
        ngpu = c.ngpu
        tag = c.tag
        path = c.path
        device = torch.device(c.device_name if(
            torch.cuda.is_available() and ngpu > 0) else "cpu")
        print(f'Using {ngpu} GPUs')
        print(device, " will be used.\n")
        cudnn.benchmark = True

        # Get train params
        l, batch_size, beta1, beta2, num_epochs, iters, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()

        # TODO read in data
        dataset = preprocess(datapath, label_mask)
        dataloader = data.DataLoader(dataset=dataset, batch_size=1)

        net = Net().to(device)

        mse_loss = nn.MSELoss(reduction='mean')
        optimizer = optim.SGD(net.parameters(), lr=lr)

        if ('cuda' in str(device)) and (ngpu > 1):
            net = nn.DataParallel(net, list(range(ngpu))).to(device)

        if not offline:
            wandb_init(tag, offline)
            wandb.watch(net, log='all', log_freq = 100)

        for epoch in range(num_epochs):
            if self.quit_flag:
                break
            times = []
            running_loss = []
            for i, d in enumerate(dataloader):
                if ('cuda' in str(device)) and (ngpu > 1):
                    start_overall = torch.cuda.Event(enable_timing=True)
                    end_overall = torch.cuda.Event(enable_timing=True)
                    start_overall.record()
                else:
                    start_overall = time.time()
                x, y = d
                # print('x shape: ', x.shape, 'y shape: ', y.shape)
                net.zero_grad()
                outputs = net(x.to(device))
                # print('outputs shape: ', outputs.shape)
                # loss = mse_loss(outputs[y != 0].view(-1, 1, 200, 200), y.to(device)[y != 0].view(-1, 1, 200, 200))
                loss = mse_loss(outputs[y != 0], y.to(device)[y != 0])
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())

            if epoch % 10 == 0:
                argmax, softmax, labels = wandb_figs(outputs.detach().cpu(), x.detach().cpu(), y.detach().cpu())
                gui_figs(outputs.detach().cpu(), x.detach().cpu(), y.detach().cpu())


                # wandb stuff - remove for final version/keep if we want it as an option
                if not offline:
                    wandb.log({'Loss': np.mean(running_loss)})
                    wandb.log({'Max Softmax Value': torch.max(outputs.detach().cpu()).detach().numpy()})
                    wandb.log({'Softmax Mean': np.mean(np.amax(outputs[0].detach().cpu().detach().numpy(), axis=0))})
                    wandb.log({'Labels': wandb.Image(labels)})
                    wandb.log({'Prediction': wandb.Image(argmax)})
                    wandb.log({'Confidence map': wandb.Image(softmax)})

                self.progress.emit(epoch, np.mean(running_loss))
                print('epoch: {}, running loss: {}'.format(epoch+1, np.mean(running_loss)))

        self.finished.emit()           
        print("TRAINING FINISHED")
