# iterative_labelling

A segmentation tool that allows you to iteratively draw labels on an image and train a CNN until you are happy with the segmentation quality.

## Folder structure

```
iterative_labelling
 ┣ src
 ┃ ┣ __init__.py
 ┃ ┣ networks.py
 ┃ ┣ postprocessing.py
 ┃ ┣ preprocessing.py
 ┃ ┣ test.py
 ┃ ┣ train.py
 ┃ ┣ train_worker.py
 ┃ ┗ util.py
 ┣ data
 ┃ ┗ example.png
 ┃ ┗ nmc_cathode.png
 ┣ .gitignore
 ┣ config.py
 ┣ main.py
 ┣ gui.py
 ┣ windows.py
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
