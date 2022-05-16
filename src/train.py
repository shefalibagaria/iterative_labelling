from src.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import tifffile
import time

def train(c, Net, offline=True, overwrite=True):
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
    dataset = preprocess(c.data_path)
    dataloader = data.DataLoader(dataset=dataset, batch_size=2)

    net = Net().to(device)

    mse_loss = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # Define Generator network
    # netG = Gen().to(device)
    # netD = Disc().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        # netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        # netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
        net = nn.DataParallel(net, list(range(ngpu))).to(device)
    # optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    # optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    if not offline:
        wandb_init(tag, offline)
        # wandb.watch(netD, log='all', log_freq=100)
        # wandb.watch(netG, log='all', log_freq=100)
        wandb.watch(net, log='all', log_freq = 100)

    for epoch in range(num_epochs):
        times = []
        running_loss = []
        for i, d in enumerate(dataloader):
            # Discriminator Training
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
            loss = mse_loss(outputs[y != 0].view(-1, 1, 200, 200), y.to(device)[y != 0].view(-1, 1, 200, 200))
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        if epoch % 10 == 0:
            f = visualise(outputs.detach().cpu(), x.detach().cpu(), y.detach().cpu())
            wandb.log({'Loss': np.mean(running_loss)})
            wandb.log({'Max Softmax Value': torch.max(outputs.detach().cpu()).detach().numpy()})
            wandb.log({'Softmax Mean': torch.mean(outputs[0].detach().cpu()).detach().numpy()})
            wandb.log({'Output': wandb.Image(f)})
            plt.close()
            print('epoch: {}, running loss: {}'.format(
                epoch+1, np.mean(running_loss)))            

    print("TRAINING FINISHED")
