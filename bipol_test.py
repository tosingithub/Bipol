import logging
from statistics import mean
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import wandb
from simpletransformers.classification import ClassificationArgs, ClassificationModel

#from utils import load_rte_data_file

jig_folder = '/home/shared_data/bipol/Jigsaw_kaggle/'
sbic_folder = '/home/shared_data/bipol/sbicv2/'
md_folder = '/home/shared_data/bipol/md/'
new_folder = '/home/shared_data/bipol/new/'
