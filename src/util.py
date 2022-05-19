import numpy as np
import torch
from torch import autograd
import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
# import torch.optim as optim
from torch.utils import data
import wandb
from dotenv import load_dotenv
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch import nn
import tifffile
from skimage.util import random_noise
from skimage import io, filters
from PIL import Image
from matplotlib import cm

# check for existing models and folders
def check_existence(tag):
    """Checks if model exists, then asks for user input. Returns True for overwrite, False for load.

    :param tag: [description]
    :type tag: [type]
    :raises SystemExit: [description]
    :raises AssertionError: [description]
    :return: True for overwrite, False for load
    :rtype: [type]
    """
    root = f'runs/{tag}'
    check_D = os.path.exists(f'{root}/Disc.pt')
    check_G = os.path.exists(f'{root}/Gen.pt')
    if check_G or check_D:
        print(f'Models already exist for tag {tag}.')
        x = input("To overwrite existing model enter 'o', to load existing model enter 'l' or to cancel enter 'c'.\n")
        if x=='o':
            print("Overwriting")
            return True
        if x=='l':
            print("Loading previous model")
            return False
        elif x=='c':
            raise SystemExit
        else:
            raise AssertionError("Incorrect argument entered.")
    return True

def check_exist(tag):
    root = f'runs/{tag}'
    if os.path.exists(f'{root}/Net.pt'):
        return True
    else:
        return False

# set-up util
def initialise_folders(tag, overwrite):
    """[summary]

    :param tag: [description]
    :type tag: [type]
    """
    if overwrite:
        try:
            os.mkdir(f'runs')
        except:
            pass
        try:
            os.mkdir(f'runs/{tag}')
        except:
            pass

def wandb_init(name, offline):
    """[summary]

    :param name: [description]
    :type name: [type]
    :param offline: [description]
    :type offline: [type]
    """
    if offline:
        mode = 'disabled'
    else:
        mode = None
    load_dotenv(os.path.join(os.getcwd(), '.env'))
    API_KEY = os.getenv('WANDB_API_KEY')
    ENTITY = os.getenv('WANDB_ENTITY')
    PROJECT = os.getenv('WANDB_PROJECT')
    if API_KEY is None or ENTITY is None or PROJECT is None:
        raise AssertionError('.env file arguments missing. Make sure WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT are present.')
    print("Logging into W and B using API key {}".format(API_KEY))
    process = subprocess.run(["wandb", "login", API_KEY], capture_output=True)
    print("stderr:", process.stderr)

    
    print('initing')
    wandb.init(entity=ENTITY, name=name, project=PROJECT, mode=mode)

    wandb_config = {
        'active': True,
        'api_key': API_KEY,
        'entity': ENTITY,
        'project': PROJECT,
        # 'watch_called': False,
        'no_cuda': False,
        # 'seed': 42,
        'log_interval': 1000,

    }
    # wandb.watch_called = wandb_config['watch_called']
    wandb.config.no_cuda = wandb_config['no_cuda']
    # wandb.config.seed = wandb_config['seed']
    wandb.config.log_interval = wandb_config['log_interval']

def wandb_save_models(fn):
    """[summary]

    :param pth: [description]
    :type pth: [type]
    :param fn: [description]
    :type fn: filename
    """
    shutil.copy(fn, os.path.join(wandb.run.dir, fn))
    wandb.save(fn)

# training util
# def preprocess(data_path):
#     """[summary]

#     :param imgs: [description]
#     :type imgs: [type]
#     :return: [description]
#     :rtype: [type]
#     """
#     img = plt.imread(data_path)[:, :, 0]
#     phases = np.unique(img)
#     if len(phases) > 10:
#         raise AssertionError('Image not one hot encoded.')
#     x, y = img.shape
#     img_oh = torch.zeros(len(phases), x, y)
#     for i, ph in enumerate(phases):
#         img_oh[i][img == ph] = 1
#     return img_oh, len(phases)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc):
    """[summary]

    :param netD: [description]
    :type netD: [type]
    :param real_data: [description]
    :type real_data: [type]
    :param fake_data: [description]
    :type fake_data: [type]
    :param batch_size: [description]
    :type batch_size: [type]
    :param l: [description]
    :type l: [type]
    :param device: [description]
    :type device: [type]
    :param gp_lambda: [description]
    :type gp_lambda: [type]
    :param nc: [description]
    :type nc: [type]
    :return: [description]
    :rtype: [type]
    """
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(
        real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

def batch_real(img, l, bs):
    """[summary]
    :param training_imgs: [description]
    :type training_imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    n_ph, x_max, y_max = img.shape
    data = torch.zeros((bs, n_ph, l, l))
    for i in range(bs):
        x, y = torch.randint(x_max - l, (1,)), torch.randint(y_max - l, (1,))
        data[i] = img[:, x:x+l, y:y+l]
    return data

# Evaluation util
def post_process(img):
    """Turns a n phase image (bs, n, imsize, imsize) into a plottable euler image (bs, 3, imsize, imsize, imsize)

    :param img: a tensor of the n phase img
    :type img: torch.Tensor
    :return:
    :rtype:
    """
    img = img.detach().cpu()
    img = torch.argmax(img, dim=1).unsqueeze(-1).numpy()

    return img * 255

def progress(i, iters, n, num_epochs, timed):
    """[summary]

    :param i: [description]
    :type i: [type]
    :param iters: [description]
    :type iters: [type]
    :param n: [description]
    :type n: [type]
    :param num_epochs: [description]
    :type num_epochs: [type]
    :param timed: [description]
    :type timed: [type]
    """
    progress = 'iteration {} of {}, epoch {} of {}'.format(
        i, iters, n, num_epochs)
    print(f"Progress: {progress}, Time per iter: {timed}")

def plot_img(img, iter, epoch, path, offline=True):
    """[summary]

    :param img: [description]
    :type img: [type]
    :param slcs: [description], defaults to 4
    :type slcs: int, optional
    """
    img = post_process(img)
    if not offline:
        wandb.log({"slices": [wandb.Image(i) for i in img]})
    else:
        fig, axs = plt.subplots(1, img.shape[0])
        for ax, im in zip(axs, img):
            ax.imshow(im)
        plt.savefig(f'{path}/{epoch}_{iter}_slices.png')
    wandb.log({"slices": [wandb.Image(i) for i in img]})
    return


def distort(img):
    """
    adds noise and blur to a segmented image to create fake raw data
    :img: a single image (np.ndarray)
    """
    img = img.astype(np.float32)
    img *= 1/np.amax(img)
    blurred = filters.gaussian(img, sigma=2)
    distorted = random_noise(blurred, mode='speckle', var=1, mean=1.5, seed=3)
    distorted = np.expand_dims(distorted, axis=0)
    return distorted

def get_window(x_size, y_size):
    #top = r.randint(0,img_stack[0].shape[0]-x_size)
    top = 100 # fix top and left for reproducability
    bottom = top+x_size
    #left = r.randint(0,img_stack[0].shape[1]-y_size)
    left = 100 # fix top and left for reproducability
    right = left+y_size
    return [left, right, top, bottom]

def crop_labels(img):
    """ Crop labelled image (so labelled info is limited)
    :img: a single image (np.ndarray)
    """
    img +=1
    coords = get_window(200,200)
    img[:,:coords[0]] = 0
    img[:,coords[1]:] = 0
    img[:coords[2]] = 0
    img[coords[3]:] = 0
    return img

def one_hot_encode(mask, n_classes=3):
    n_classes+=1
    one_hot = np.zeros((n_classes, mask.shape[0], mask.shape[1]))
    for i, unique_value in enumerate(np.unique(mask)):
        one_hot[i][mask == unique_value] = 1
    one_hot = one_hot[1:]   # remove '0' (unlabelled) layer
    return one_hot

# def preprocess(data_path):
#     """
#     :data_path: path to data in tif format (string)
#     """
#     imgs = io.imread(data_path)
#     inputs = []
#     targets = []
#     for img in imgs:
#         inputs.append(distort(img))
#         cropped = crop_labels(img)
#         targets.append(one_hot_encode(cropped))
#     dataset = NMCDataset(inputs=inputs, targets=targets, transform=None)
#     return dataset

def preprocess(data_path, labels):
    # function returns dataset to be trained
    inputs = []
    targets = []
    imgs = io.imread(data_path, as_gray=True)
    if data_path[-3:] =='tif':
        for img in imgs:
            inputs.append(img)
    else:
        img = io.imread(data_path, as_gray = True)
        inputs = [np.expand_dims(img, axis=0)]
    targets = [labels]
    dataset = NMCDataset(inputs = inputs, targets = targets, transform = None)
    return dataset

def visualise(output, x, y):
    """
    :output: one-hot encoded output from network (torch.Tensor, shape = [batch, classes, height, width])
    :x: input into vector (greyscale or rgb) (torch.Tensor, shape = [batch, channels, height, width])
    :y: one-hot encoded labelled pixels (torch.Tensor, shape = [batch, classes, height, width])
    """
    img_stack = io.imread("data/3ph_0/NMC_90wt_0bar.tif")

    titles = ['input', 'ground truth', 'mask','argmax', 'output as rgb',
              'max of softmax']
    input = x[0,0]
    ground_truth = img_stack[9]
    mask = torch.argmax(y,dim=1)[0]
    argmax = torch.argmax(output, dim=1)[0]
    rgb = output[0].permute(1,2,0).detach().numpy()
    softmax_max = torch.max(output[0], 0)[0].detach().numpy()
    imgs = [input, ground_truth, mask, argmax, rgb, softmax_max]
    fig, ax = plt.subplots(2, 3, figsize=[20,12])
    with torch.no_grad():
        for i in range(6):
            x_coord = int(i/3) # for i<3 x=0, for i<=3 x=1 
            y_coord = i%3
            if i>3:
                im = ax[x_coord,y_coord].imshow(imgs[i], vmin=0, vmax = 1)
            else:
                im = ax[x_coord,y_coord].imshow(imgs[i])
            ax[x_coord,y_coord].axis('off')
            ax[x_coord,y_coord].title.set_text(titles[i])
            divider = make_axes_locatable(ax[x_coord,y_coord])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)
        # plt.show()
    return fig

def wandb_figs(output, x, y):
    output = output.numpy()
    argmax = 1 + np.argmax(output, axis=1)[0]
    argmax_norm = argmax/np.amax(argmax)
    img_argmax = Image.fromarray(np.uint8(cm.inferno(argmax_norm)*255))

    softmax_max = np.amax(output, axis=1)[0]
    img_softmax = Image.fromarray(np.uint8(cm.inferno(softmax_max)*255))

    layer = np.zeros((1, output.shape[2], output.shape[3]))
    labels = np.argmax(np.concatenate((layer, y.numpy()[0]), axis=0), axis=0)
    labels_norm = labels/np.amax(labels)
    img_labels = Image.fromarray(np.uint8(cm.inferno(labels_norm)*255))

    return img_argmax, img_softmax, img_labels

def gui_figs(output, x, y):
    output = output.numpy()
    x = x.numpy()

    n_colours = 9
    n_classes = output.shape[1]

    argmax = np.argmax(output, axis=1)[0]
    argmax_norm = 1/18+(argmax/n_colours)
    im_argmax = Image.fromarray(np.uint8(cm.Set1(argmax_norm)*255), mode='RGBA')
    im_argmax.save('data/temp/prediction.png')

    softmax = np.amax(output, axis=1)[0]
    softmax = (softmax-(1/n_classes))*(n_classes/(n_classes-1))
    # print(f'n classes: {n_classes}, softmax max: {np.amax(softmax)}, softmax min: {np.amin(softmax)}')
    alpha = 0.8*(1-softmax)
    alpha = np.expand_dims(alpha, axis=2)
    black = np.zeros((output.shape[2], output.shape[3], 3))
    im_softmax = Image.fromarray(np.uint8(np.concatenate((black, alpha), axis=2)*255), mode='RGBA')
    im_softmax.save('data/temp/confidence.png')

    inputs = x[0][0]
    im_inputs = Image.fromarray(np.uint8(inputs*255), mode='L')
    im_inputs = im_inputs.convert('RGBA')
    im_inputs.save('data/temp/inputs.png')

    blended_cp = Image.alpha_composite(im_argmax, im_softmax)
    blended_cp.save('data/temp/confidence_prediction.png')

    im_argmax.putalpha(110)
    blended_p = Image.alpha_composite(im_inputs, im_argmax)
    blended_p.save('data/temp/prediction_blend.png')

    blended_c = Image.alpha_composite(im_inputs, im_softmax)
    blended_c.save('data/temp/confidence_blend.png')
    return
    

    




class NMCDataset(data.Dataset):
    def __init__(self, inputs: list, targets: list, transform=None):
        self.inputs = inputs
        self.targets = targets
        self. transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        x = self.inputs[index]
        y = self.targets[index]

        if self.transform is not None:
            x, y = self.transform(x, y)
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        return x, y
