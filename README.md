# iterative_labelling

A segmentation tool that allows you to iteratively draw labels on an image and train a CNN until you are happy with the segmentation quality.

## Folder structure

```
iterative_labelling
 ┣ src
 ┃ ┣ __init__.py
 ┃ ┣ networks.py - build network using confid settings
 ┃ ┣ train_worker.py - worker for training network in a seprate thread
 ┃ ┗ util.py - data proceesing functions
 ┣ data
 ┃ ┗ nmc_cathode.png - uncropped NMC cathode XCT image slice
 ┃ ┗ nmc_cathode_512.png - 512x512 raw image used for testing
 ┃ ┗ nmc_cathode_128.png - 128x128 image used in manual labelling test
 ┃ ┗ synthetic_x.png - distorted synthetic data
 ┃ ┗ synthetic_y.png - segmented data used as ground truth
 ┣ runs - folder contains data from user tests and synthetic data test
 ┣ .gitignore
 ┣ config.py - network configuration
 ┣ main.py
 ┣ gui.py - start GUI to label image & train network
 ┣ windows.py - Visualise and options windows
 ┣ analyse_results.ipynb - notebook where results from user tests and network validation were analysed
 ┣ README.md
 ┗ requirements.txt
```

## Quickstart

Prerequisites:

- pip3
- python3

Navigate to the repo folder. Create a python venv, activate, install pytorch and required libraries from requirements.txt

_Note: cudatoolkit version and pytorch install depends on system, see [PyTorch install](https://pytorch.org/get-started/locally/) for more info._

```
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision
pip3 install -r requirements.txt
```

If you want to log your runs on Weights and Biases, create a .env file to hold secrets, the .env file must include

```
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=
```

You are now ready to run the repo. To start training

```
python3 gui.py
```

To track your runs with WandB, make sure to select this in the options dialog.
