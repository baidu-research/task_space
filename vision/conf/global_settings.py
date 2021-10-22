""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
