import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from lib.pipeline.utils import list_subjects_folders, list_folder


def load_subjects_data(path, source, use_raw_data):
    x = {}
    y = {}
    
    for subject in list_subjects_folders(path):
        subject_dir = os.path.join(path, subject)
        
        x[subject] = np.load(os.path.join(subject_dir, f'{subject}_{source}{"_features" if not use_raw_data else ""}.npy'))
        y[subject] = np.load(os.path.join(subject_dir, f'{subject}_{source}_gt.npy'))
    
    return x, y


def ground_truth_to_categorical(y, mapping):
    y_copy = y.copy()
    for subject, gt in y_copy.items():
        mapped_gt = list(map(lambda i : mapping[i], gt))
        y_copy[subject] = to_categorical(mapped_gt, len(mapping))
        
    return y_copy


def load_data(path, source, use_raw_data, gt_mapping):
    x, y = load_subjects_data(path, source, use_raw_data)
    y = ground_truth_to_categorical(y, gt_mapping)
    
    return x, y


def load_dataset(path):
    '''
    Loads the accelerometer and gyroscope data from dataset.
    
    Args:
        path (str): Root directory of the data.
        
    Returns:
        data (dict): Dict containing pandas dataframes with the accelerometer and gyroscope data for each execution.
    '''
    
    subjects = list_subjects_folders(path)
    data = {}

    for subject in subjects:        
        subject_dir = os.path.join(path, subject)
        subject_files = list_folder(subject_dir)

        for file in subject_files:
            file_path = os.path.join(subject_dir, file)
            file_desc = file.split('.')[0]
            if not os.path.isfile(file_path) or not file_path.endswith('.csv'):
                continue

            data[file_desc] = pd.read_csv(file_path)
    
    return data
