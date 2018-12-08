import os
import argparse
import glob
from random import shuffle
import numpy as np
from tqdm import tqdm

from utils.feature_ext import YaafeMFCC

"""
Preprocessing audio files (.wav-files) for speaker identification. The files are assumed
to lie in a folder of following structure:
- one root folder
- for each speaker in the dataset the root folder has one subdirectory holding the .wav-files
  belonging to that speaker
- subdirectories themself can have several subdirectories

The data will be preprocessed and parsed into folder with two subfolder - classification and identification:
- classification: random small minibatches of size min_size which contain all speaker
- identification: has two subdirectories (train and val). train and val contain different speaker

The reason for small randomly assembled units (.npy files) is to later efficiently load the data into 
cache during training.
"""

# parse args
parser = argparse.ArgumentParser(description='Preprocessing audio files (.wav-files). Computing MFCC features,' +  
                                 'normalizing and creating small random minibatches')
parser.add_argument('data_root', type=str, help='root folder of dataset')
parser.add_argument('data_dest', type=str, help='root folder of preprocessed data')
parser.add_argument('--window_length', default=80, type=int, help='number of neighboring MFCC features for each data point')
parser.add_argument('--window_step', default=20, type=int, help='step size for extracting the windows of MFCC features from data')
parser.add_argument('--min_size', default=256, type=int, help='number samples in small mini_batch (must be <= batch_size)')
parser.add_argument('--val_split', default=0.2, type=float, help='portion of the validation set')
args = parser.parse_args()

def run_feature_extraction(path_targ: list) -> np.array:
    """
    creates 35 MFCC features per 25 ms in this cofiguration and initializes target vector
    
    args:
        path_targ: list: first element path to audio (.wav) file, second element target
    returns:
        features: np array with dim (x, 35) where x is the number "duration" with time step
                "step" fits into the autio file
    """
    
    path, target = path_targ
    
    feature_extractor =  YaafeMFCC(**{ "duration":0.025,
                                      "step":0.010,
                                      "stack":1,
                                      "e":False,
                                      "coefs":11,
                                      "De":True,
                                      "DDe":True,
                                      "D":True,
                                      "DD":True})
    
    features = feature_extractor(path)
    target = int(target)

    return features, target

def normalize_batch(batch: np.array) -> np.array:
    """
    Normalizes a batch of size N, W, H
    """
    mean = batch.mean(axis=(1,2))
    std = batch.std(axis=(1,2))
    N, W, H = batch.shape
    return np.reshape((np.reshape(batch, (W,H,N)) - mean) / std, (N, W, H))

def get_all_file_paths_labels(data_root: str) -> list:
    """
    Gets the paths of all wav-files in the data root directory plus there labels
    
    args:
        data_root: string holding name of root directory
    returns:
        all_files: list of lists. Each sub list contains two elements: path to file and label of that file
    """
    
    speaker_dirs = os.listdir(data_root)
    all_files = []
    i = 0
    for d in speaker_dirs:
        files = glob.iglob(data_root + '/' + d + '/**/*.wav', recursive=True)
        files = [[f, i] for f in files]
        all_files += files
        i+=1
    return all_files

def split_all_files_val_classification(all_files: str) -> list:
    
    # get all speaker labels
    labels = np.unique([f[1] for f in all_files])
    
    train_files = []
    val_files = []
    for l in labels:
        l_files = [f for f in all_files if f[1]==l]
        v_idx = np.random.choice(np.arange(len(l_files)), size=int(len(l_files)*args.val_split), replace=False)
        v_files = [l_files[i] for i in range(len(l_files)) if i in v_idx]  
        t_files = [l_files[i] for i in range(len(l_files)) if i not in v_idx]
        train_files += t_files
        val_files += v_files
        
    return train_files, val_files

def get_sequences_of_samples(features: np.array) -> np.array:
    """
    Creates sequences of features of the input
    
    args:
        features: np.array holding a sequence of MFCC features (shape: T, F )
    returns:
        samples: np.array holding several sub sequences of features (shape: ((T-)-T%step)/step, length, F)
    """
    
    length = args.window_length
    step = args.window_step
    N = features.shape[0]
    
    if N <= length:
        return None
    
    lower_idx = 0
    upper_idx = length
    samples = []
    while upper_idx < N:
        samples.append(features[lower_idx:upper_idx])
        lower_idx += step
        upper_idx += step
    if upper_idx < N-1:
        samples.append(features[-length:])
    samples = np.array(samples)
    
    return samples

def fill_samples_cache(sample_cache: np.array, sample_cache_targets: np.array, files: list, msize: int=10000):
    """
    Refills cache of samples for filling mini batch files and removes corresponding files from file list.
    If sample_cache and sample_cache_targets are None the cache is initialized.
    
    args:
        sample_cache: current cache of samples
        sample_cache_targets: current cache of targets
        files: list of file names and their targets
        msize: int. min size of refilled cache
    returns:
        sample_cache: updated cache of samples
        sample_cache_targets: updated cache of targets
        files: shrinked list of file names and their targets
    """

    shuffle(files)
    if sample_cache is None:
        sample_count = 0
    else:
        sample_count = sample_cache.shape[0]
    new_cache_samples = []
    new_cache_samples_targets = []
    while sample_count < msize and len(files) > 0:
        path = files.pop(0)
        s = run_feature_extraction(path)
        if s is not None:
            new_samples, new_samples_target = s[0], s[1]
            new_samples = get_sequences_of_samples(new_samples)
            if new_samples is not None:
                new_samples = normalize_batch(new_samples)
                new_samples_targets = np.ones(new_samples.shape[0]) * new_samples_target
                new_cache_samples.append(new_samples)
                new_cache_samples_targets.append(new_samples_targets)
                sample_count += new_samples.shape[0]
    
    new_cache_samples = np.concatenate(new_cache_samples, axis=0)
    new_cache_samples_targets = np.concatenate(new_cache_samples_targets, axis=0)
    if sample_cache is None:
        sample_cache = new_cache_samples
        sample_cache_targets = new_cache_samples_targets
    else:
        sample_cache = np.concatenate([new_cache_samples, sample_cache], axis=0)
        sample_cache_targets = np.concatenate([new_cache_samples_targets, sample_cache_targets], axis=0)
    
    return sample_cache, sample_cache_targets, files

def create_set(root_dest: str, files: list):
    """
    Creating a set of randomly assembled mini batch files each containing min_size sequences of MFCC features of length
    
    args:
        root_dest: string holding destination folder
        fiels: list of lists. Each sub list contains two elements: path to file and label of that file
    returns:
        None
    """
    
    min_batch_size = args.min_size
    
    # initialize caches
    samples_cache, samples_cache_targets, files = fill_samples_cache(None, None, files)
    i = 0
    while len(files) > 0 or samples_cache.shape[0] >= min_batch_size:
        
        print('Remaining audio files: ' + str(len(files)))
        
        idxs = np.arange(samples_cache.shape[0])
        shuffle(idxs)
        samples_cache = samples_cache[idxs]
        samples_cache_targets = samples_cache_targets[idxs]

        while samples_cache.shape[0] >= min_batch_size:
            print('Dumping ' + (root_dest + '/' + str(i) + '.npy'))
            np.save(root_dest + '/' + str(i) + '.npy', samples_cache[:min_batch_size])
            np.save(root_dest + '/' + str(i) + '_targets.npy', samples_cache_targets[:min_batch_size])
            samples_cache = samples_cache[min_batch_size:]
            samples_cache_targets = samples_cache_targets[min_batch_size:]
            i += 1
        
        if len(files) > 0:
            samples_cache, samples_cache_targets, files = fill_samples_cache(samples_cache, 
                                                                             samples_cache_targets, files)
    

def preprocess_data():
    """
    main function for executing preprocessing flow
    """
    
    # get all .wav files and their lables
    data_root = args.data_root
    assert os.path.exists(data_root)
    all_files = get_all_file_paths_labels(data_root)
    
    # create directories
    data_dest = args.data_dest 
    if not os.path.exists(data_dest):
        os.mkdir(data_dest)
    classification_root = data_dest + '/classification'
    identification_root = data_dest + '/identification'
    os.mkdir(classification_root)
    os.mkdir(classification_root + '/train')
    os.mkdir(classification_root + '/val')
    os.mkdir(identification_root)
    os.mkdir(identification_root + '/train')
    os.mkdir(identification_root + '/val')
    
    # create classification dataset
    train_files_class, val_files_class = split_all_files_val_classification(all_files.copy())
    print('Creating classification training set...')
    create_set(classification_root + '/train', train_files_class.copy())
    print('Creating classification validation set...')
    create_set(classification_root + '/val', val_files_class.copy())
    
    # create identification dataset
    speaker = np.unique([f[1] for f in all_files])
    N_speaker = len(speaker)
    val_idx = np.random.choice(np.arange(N_speaker), size=int(N_speaker*args.val_split), replace=False)
    train_files = [f for f in all_files if f[1] not in val_idx]
    val_files = [f for f in all_files if f[1] in val_idx]
    print('Creating identification training dataset...')
    create_set(identification_root + '/train', train_files.copy())
    print('Creating identification validation dataset...')
    create_set(identification_root + '/val', val_files.copy())
    
    print('Done!')

if __name__ == '__main__':
    preprocess_data()
