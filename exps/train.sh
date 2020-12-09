pip install -U requirements.txt
pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer.git

arg = wandb sweep sweep.yaml

wandb agent $arg