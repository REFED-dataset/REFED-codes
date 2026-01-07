import argparse
from datetime import datetime

'''
This file is used to define and parse command line arguments for training the TSMMF_REFED model.
'''

def get_args():
    parser = argparse.ArgumentParser('Train_TSMMF_REFED')
    
    # data and task related
    parser.add_argument('--path_data',    type=str,   default="REFED-dataset/preprocessed/")
    parser.add_argument('--path_label',   type=str,   default="REFED-dataset/annotations/")
    parser.add_argument('--modality',     type=str,   default="both") # 'EEG','fNRIS','both'
    parser.add_argument('--num_trial',    type=int,   default= 15)
    parser.add_argument('--label_dim',    type=str,   default="valence") # 'valence', 'arousal'
    parser.add_argument('--label_mode',   type=str,   default="3c") # '3c', '0-1', '-1-1'
    '''
    label modes:
        3c:   low, medium, high classification
        0-1:  [0,1] regression, netural is 0.5
        -1-1: [-1,1] regression, neutral is 0.0
    '''
    
    # training setting
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument('--cuda',         type=str,   default='0')
    parser.add_argument('--n_epoch',      type=int,   default=100)
    parser.add_argument('--patience',     type=int,   default=20)
    parser.add_argument('--optimizer',    type=str,   default='Adam')
    parser.add_argument('--lr',           type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--load_workers', type=int,   default=0)
    parser.add_argument('--final_activation', type=str, default='None') # tanh, sigmoid, None
    # 'None': for classification, softmax is included in loss function
    # 'tanh': for regression in [-1,1]
    # 'sigmoid': for regression in [0,1]
    
    args = parser.parse_args()
    args.final_activation = None if args.final_activation == 'None' else args.final_activation
    return args
