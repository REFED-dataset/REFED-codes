import os
from scipy.io import loadmat
import gc


def load_data(data_path, sub_list=None, modality:list=['EEG', 'fNIRS']):
    '''
    load EEG and fNIRS data from REFED dataset
    Inputs:
        data_path: str, the path of REFED dataset
        sub_list: list of str, the subject list to be loaded
        modality: list of str, the modality to be loaded, options are 'EEG' and 'fNIRS'
    Outputs:
        data: dict, the loaded data, data[subject_id][modality]['v1'~'v15']
    '''
    data = {}
    if sub_list is None:
        sub_list = os.listdir(data_path)
    
    for si in sub_list:
        # Determine whether it is the subject folder
        if os.path.isdir(os.path.join(data_path, si)):
            data[si] = {}
            # Read EEG and F-NIRS data
            if 'EEG' in modality:
                path_si_EEG = os.path.join(data_path, si, 'EEG_videos.mat')
                data_si_EEG = loadmat(path_si_EEG)
                data[si]['EEG'] = {'v%d'%vi :data_si_EEG['video_%d' % vi] for vi in range(1,16)}
                del data_si_EEG
            if 'fNIRS' in modality:
                path_si_fNIRS = os.path.join(data_path, si, 'fNIRS_videos.mat')
                data_si_fNIRS = loadmat(path_si_fNIRS)
                data[si]['fNIRS'] = {'v%d'%vi:data_si_fNIRS['video_%d' % vi] for vi in range(1,16)}
                del data_si_fNIRS
            gc.collect()
    return data



def load_baseline(data_path, sub_list=None, modality:list=['EEG', 'fNIRS']):
    '''
    load EEG and fNIRS baseline data from REFED dataset
    Inputs:
        data_path: str, the path of REFED dataset
        sub_list: list of str, the subject list to be loaded
        modality: list of str, the modality to be loaded, options are 'EEG' and 'fNIRS'
    Outputs:
        data: dict, the loaded data, data[subject_id][modality]['v1'~'v15']
    '''
    data = {}
    if sub_list is None:
        sub_list = os.listdir(data_path)
    
    for si in sub_list:
        # Determine whether it is the subject folder
        if os.path.isdir(os.path.join(data_path, si)):
            data[si] = {}
            # Read EEG and F-NIRS data
            if 'EEG' in modality:
                path_si_EEG = os.path.join(data_path, si, 'EEG_baselines.mat')
                data_si_EEG = loadmat(path_si_EEG)
                data[si]['EEG'] = {'v%d'%vi :data_si_EEG['video_%d' % vi] for vi in range(1,16)}
                del data_si_EEG
            if 'fNIRS' in modality:
                path_si_fNIRS = os.path.join(data_path, si, 'fNIRS_baselines.mat')
                data_si_fNIRS = loadmat(path_si_fNIRS)
                data[si]['fNIRS'] = {'v%d'%vi:data_si_fNIRS['video_%d' % vi] for vi in range(1,16)}
                del data_si_fNIRS
            gc.collect()
    return data



def load_feature(data_path, sub_list=None, modality:list=['EEG', 'fNIRS']):
    '''
    load EEG and fNIRS feature data from REFED dataset
    Inputs:
        data_path: str, the path of REFED dataset
        sub_list: list of str, the subject list to be loaded
        modality: list of str, the modality to be loaded, options are 'EEG' and 'fNIRS'
    Outputs:
        data: dict, the loaded data, data[subject_id][modality]['v1'~'v15']
    '''
    data = {}
    if sub_list is None:
        sub_list = os.listdir(data_path)
    
    for si in sub_list:
        # Determine whether it is the subject folder
        if os.path.isdir(os.path.join(data_path, si)):
            data[si] = {}
            # Read EEG and F-NIRS data
            if 'EEG' in modality:
                path_si_EEG = os.path.join(data_path, si, 'EEG_videos_feature.mat')
                data_si_EEG = loadmat(path_si_EEG)
                data[si]['EEG'] = {'v%d'%vi :data_si_EEG['video_%d' % vi] for vi in range(1,16)}
                del data_si_EEG
            if 'fNIRS' in modality:
                path_si_fNIRS = os.path.join(data_path, si, 'fNIRS_videos_feature.mat')
                data_si_fNIRS = loadmat(path_si_fNIRS)
                data[si]['fNIRS'] = {'v%d'%vi:data_si_fNIRS['video_%d' % vi] for vi in range(1,16)}
                del data_si_fNIRS
            gc.collect()
    return data



def load_label(data_path, sub_list=None, dimension:list=['Valence', 'Arousal']):
    '''
    load label from REFED dataset
    Inputs:
        data_path: str, the path of REFED dataset
        sub_list: list of str, the subject list to be loaded
        dimension: list of str, the dimension to be loaded, options are 'Valence' and 'Arousal'
    Outputs:
        label: dict, the loaded label, label[subject_id]['v1'~'v15']['Valence'/'Arousal']
    '''
    label = {}
    if sub_list is None:
        sub_file = os.listdir(data_path)
    else:
        sub_file = ['s%s_label.mat' % si for si in sub_list]
        
    for si in sub_file:
        if si.endswith('_label.mat'):
            si_key = 's%s' % si[:-10]
            file_path = os.path.join(data_path, si)
            label_si = loadmat(file_path)
            
            label[si_key] = {}
            for vi in range(1,16):
                vi_key = 'v%d' % vi
                label[si_key][vi_key] = {}
                if 'Valence' in dimension:
                    label[si_key][vi_key]['Valence'] = label_si['video_%d' % vi][:, 0]
                if 'Arousal' in dimension:
                    label[si_key][vi_key]['Arousal'] = label_si['video_%d' % vi][:, 1]
    return label
