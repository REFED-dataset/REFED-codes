import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from utils.load_REFED import load_data as load_REFED_data, load_label as load_REFED_label


def load_data(idx, path_data, path_label, modality=['EEG', 'fNIRS'], info=['video', 'baseline', 'label', 'time']):
    '''
    load data for one subject
    Input:
        idx: subject id
        path_data: path to data
        path_label: path to label
        modality: list, modalities to load
        info: list, info to load
    Output:
        data: dict, loaded data
    '''
    data = {}
    if 'EEG' in modality:
        read_data = load_REFED_data(path_data, idx, modality=['EEG'])[idx]['EEG']
        data['EEG'] = [read_data[vi] for vi in read_data.keys()]
        if 'baseline' in info:
            read_data = load_REFED_data(path_data, idx, modality=['EEG'])[idx]['EEG']
            data['EEG_baseline'] = [read_data[vi] for vi in read_data.keys()]
    if 'fNIRS' in modality:
        read_data = load_REFED_data(path_data, idx, modality=['fNIRS'])[idx]['fNIRS']
        data['fNIRS'] = [read_data[vi][:,2] for vi in read_data.keys()]
        if 'baseline' in info:
            read_data = load_REFED_data(path_data, idx, modality=['fNIRS'])[idx]['fNIRS']
            data['fNIRS_baseline'] = [read_data[vi][:,2] for vi in read_data.keys()]
    if 'label' in info:
        read_data = load_REFED_label(path_label, idx)[idx]
        data['label'] = [np.stack([read_data[vi]['Valence'], read_data[vi]['Arousal']],axis=1) for vi in read_data.keys()]
        if 'time' in info:
            data['time'] = [np.arange(len(l)) for l in data['label'] ]
    return data


def process_data(data, label='3c'):
    '''
    Process data: add channel dimension, resampling, baseline correction
    Input:
        data: dict, raw data
        label: str, label mode
    Output:
        data: dict, processed data
    '''
    if 'EEG' in data:
        data['EEG'] = [np.expand_dims(data['EEG'][vi], axis=1) for vi in range(len(data['EEG']))]
    if 'fNIRS' in data:
        data['fNIRS'] = [np.expand_dims(data['fNIRS'][vi], axis=1) for vi in range(len(data['fNIRS']))]
        
    # You can add more processing steps here if needed, such as resampling or filtering
    
    return data


def process_label(data, dimension, label_mode='3c'):
    '''
    Process label: convert to specified mode and dimension
    Input:
        data: dict, raw data
        dimension: str, 'valence', 'arousal', 'both'
        label_mode: str, '3c', '0-1', '-1-1'
    Output:
        data: dict, processed data
    '''
    for vi in range(len(data['label'])):
        if 'label' in data:
            if label_mode == '3c':
                # convert to 3 categories
                data['label'][vi] = label_to_3c(data['label'][vi])
            elif label_mode == '0-1':
                # convert to 0-1 scale
                data['label'][vi] = (data['label'][vi]-1)/254
            elif label_mode == '-1-1':
                # convert to -1-1 scale
                data['label'][vi] = (data['label'][vi]-1)/127-1
            else:
                print("label type error")
            if dimension == 'valence':
                data['label'][vi] = data['label'][vi][:,0]
            elif dimension == 'arousal':
                data['label'][vi] = data['label'][vi][:,1]
            elif dimension != 'both':
                print("dimension type error")
    return data


# Labels are converted into 3 categories
def label_to_3c(label, thresholds=[0.3, 0.7], original_scale=256):
    '''
    Convert continuous labels to 3 categories: 0 (low), 1 (medium), 2 (high)
    Input:
        label: np.array, continuous labels
        thresholds: list, thresholds for low and high categories
        original_scale: int, original scale of the labels
    Output:
        label_new: np.array, categorical labels
    '''
    label_new = np.ones_like(label)
    label_new[label < original_scale*thresholds[0]] = 0
    label_new[label > original_scale*thresholds[1]] = 2
    return label_new


# Labels are converted into 3 categories with 0-1 or -1-1 scale
def label_to_3c_01(label, mode='0-1'):
    '''
    Convert continuous labels to 3 categories: 0 (low), 1 (medium), 2 (high)
    Input:
        label: np.array, continuous labels
        mode: str, '0-1' or '-1-1'
    Output:
        label_new: np.array, categorical labels
    '''
    label_new = np.ones_like(label)
    if mode == '0-1':
        label_new[label<=0.3] = 0
        label_new[label>=0.7] = 2
    elif mode == '-1-1':
        label_new[label<=-0.4] = 0
        label_new[label>=0.4] = 2
    return label_new


def get_fold_data(data, fold_idx, label_mode, val_ratio=0.2):
    '''
    Get train, validation, and test data for a specific fold
    Input:
        data: dict, processed data
        fold_idx: int, index of the fold to be used as test set
        label_mode: str, label mode
        val_ratio: float, ratio of validation set in training data
    Output:
        data_train: dict, training data
        data_valid: dict, validation data
        data_test: dict, test data
    '''
    train_EEG = np.concatenate([data['EEG'][i] for i in range(len(data['EEG'])) if i != fold_idx], axis=0)
    train_fNIRS = np.concatenate([data['fNIRS'][i] for i in range(len(data['fNIRS'])) if i != fold_idx], axis=0)
    train_label = np.concatenate([data['label'][i] for i in range(len(data['label'])) if i != fold_idx], axis=0)
    test_EEG = data['EEG'][fold_idx]
    test_fNIRS = data['fNIRS'][fold_idx]
    test_label = data['label'][fold_idx]
    if label_mode == '3c':
        # Stratified split for validation set
        try:
            # StratifiedShuffleSplit
            ss_split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio)
            for train_index, val_index in ss_split.split(train_EEG, train_label):
                val_EEG, val_fNIRS, val_label = train_EEG[val_index], train_fNIRS[val_index], train_label[val_index]
                train_EEG, train_fNIRS, train_label = train_EEG[train_index], train_fNIRS[train_index], train_label[train_index]
                break
        except:
            # Fallback to random shuffle if StratifiedShuffleSplit fails
            print("StratifiedShuffleSplit failed, use random shuffle instead.")
            idx = np.arange(len(train_label))
            np.random.shuffle(idx)
            val_size = int(len(train_label)*val_ratio)
            val_idx, train_idx = idx[:val_size], idx[val_size:]
            val_EEG, val_fNIRS, val_label = train_EEG[val_idx], train_fNIRS[val_idx], train_label[val_idx]
            train_EEG, train_fNIRS, train_label = train_EEG[train_idx], train_fNIRS[train_idx], train_label[train_idx]
    else:
        # Random shuffle for validation set
        idx = np.arange(len(train_label))
        np.random.shuffle(idx)
        val_size = int(len(train_label)*val_ratio)
        val_idx, train_idx = idx[:val_size], idx[val_size:]
        val_EEG, val_fNIRS, val_label = train_EEG[val_idx], train_fNIRS[val_idx], train_label[val_idx]
        train_EEG, train_fNIRS, train_label = train_EEG[train_idx], train_fNIRS[train_idx], train_label[train_idx]

        train_label = np.expand_dims(train_label, -1)
        val_label = np.expand_dims(val_label, -1)
        test_label = np.expand_dims(test_label, -1)
    
    # Return data
    data_train = {'EEG': train_EEG, 'fNIRS': train_fNIRS, 'label': train_label}
    data_valid = {'EEG': val_EEG, 'fNIRS': val_fNIRS, 'label': val_label}
    data_test = {'EEG': test_EEG, 'fNIRS': test_fNIRS, 'label': test_label}
    return data_train, data_valid, data_test


class EEG_fNIRS_Dataset(Dataset):
    '''
    EEG-fNIRS Dataset
    '''
    def __init__(self, eeg, fnirs, y, label_mode):
        assert len(eeg) == len(fnirs), \
            'The number of EEG(%d) and fNIRS(%d) does not match.' % (len(eeg), len(fnirs))
        assert len(eeg) == len(y), \
            'The number of EEG(%d) and label(%d) does not match.' % (len(eeg), len(y))

        self.eeg = eeg.astype(np.float32)
        self.fnirs = fnirs.astype(np.float32)
        if label_mode == '3c':
            self.y = y.astype(np.int64)
        else:
            self.y = y.astype(np.float32)
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.eeg[index], self.fnirs[index], self.y[index]
        
