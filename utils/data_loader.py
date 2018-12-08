import numpy as np
import keras.backend as K
from keras.utils import to_categorical


def data_generator_classification(npy_files: list, batch_size: int, steps: int, num_classes: int,
                                  mode: str='train', model: str='CNN'):
    """
    Generator for generating training and validation data for the speaker classification training.
    
    args:
        npy_files: list of lists. Each sublist consist of two elements. The first is the path to the npy file 
                    with the features and the secon with path to the targets
        batch_size: int with batch size
        steps: int meaning number of batches per epoch
        mode: string indicating whether this generator is used for training or validation data
        model: string holding type of model we are training (only CNN or LSTM) for adding channels in case of CNN
        
    yields:
        (features, targets): tuple with feature (shape: LSTM -> batch_size, W, H; CNN -> batch_size, W, H, 1) 
                            and target batch 
    """
    
    # load sample file to determine shapes of the data
    f = np.load(npy_files[0][0])
    f_N, f_W, f_H = f.shape
    
    idxs = np.arange(len(npy_files))
    n = int(batch_size/f_N)
    assert mode=='train' or mode=='val'
    
    if mode == 'train':
        while True:
            
            # get files and retrieve data
            f_idxs = np.random.choice(idxs, size=n)
            features = np.zeros((batch_size, f_W, f_H))
            targets = np.zeros(batch_size)
            for i in range(n):
                file = npy_files[f_idxs[i]]
                dfile = file[0]
                tfile = file[1]
                samples = np.load(dfile)
                sample_targets = np.load(tfile)
                features[i*f_N:(i+1)*f_N] = samples
                targets[i*f_N:(i+1)*f_N] = sample_targets

            # transform targets to categrorical
            targets = to_categorical(targets, num_classes)
            
            # if CNN is trained insert channel
            if model == 'CNN':
                B, W, H = features.shape
                if K.image_data_format() == 'channels_last':
                    features = np.reshape(features, (B, W, H, 1))
                else:
                    features = np.reshape(features, (B, 1, W, H))

            assert features.shape[0] == batch_size

            yield (features, targets)
        
    # for val mode we dont need to draw random files, but just loop over the data
    if mode == 'val':
        while True:
            
            # get files and retrieve data
            for j in range(steps):
                features = np.zeros((batch_size, f_W, f_H))
                targets = np.zeros(batch_size)
                for i in range(n):
                    file = npy_files[j+i]
                    dfile = file[0]
                    tfile = file[1]
                    samples = np.load(dfile)
                    sample_targets = np.load(tfile)
                    features[i*f_N:(i+1)*f_N] = samples
                    targets[i*f_N:(i+1)*f_N] = sample_targets

                # transform targets to categrorical 
                targets = to_categorical(targets, num_classes)

                # if CNN is trained insert channel
                if model == 'CNN':
                    B, W, H = features.shape
                    if K.image_data_format() == 'channels_last':
                        features = np.reshape(features, (B, W, H, 1))
                    else:
                        features = np.reshape(features, (B, 1, W, H))

                assert features.shape[0] == batch_size

                yield (features, targets)
                

def data_generator_identification(npy_files, batch_size, steps, cache_size=2, mode='train', model='CNN'):
    """
    Generator for generating training and validation data for the speaker identification training.
    
    args:
        npy_files: list of lists. Each sublist consist of two elements. The first is the path to the npy file 
                    with the features and the secon with path to the targets
        batch_size: int with batch size
        steps: int meaning number of batches per epoch
        mode: string indicating whether this generator is used for training or validation data
        model: string holding type of model we are training (only CNN or LSTM) for adding channels in case of CNN
        
    yields:
        (features, targets): tuple with feature (shape: LSTM -> batch_size, W, H; CNN -> batch_size, W, H, 1) 
                            and target batch 
    """
    
    # load sample file to determine shapes of the data
    f = np.load(npy_files[0][0])
    f_N, f_W, f_H = f.shape
    
    idxs = np.arange(len(npy_files))
    n = int(batch_size/f_N)
        
    # get labels
    labels = []
    for file in npy_files:
        targets = np.load(file[1])
        labels = list(np.unique(labels + list(np.unique(targets))))
    labels = [int(i) for i in labels]
    
    # build positive sample cache
    cache_pos = dict()
    for i in labels:
        cache_pos[i] = []

    cache_not_full = True
    while cache_not_full:

        f_idxs = np.random.choice(idxs, size=n)

        for i in range(n):
            file = npy_files[f_idxs[i]]
            dfile = file[0]
            tfile = file[1]
            sample_targets = np.load(tfile)
            samples = np.load(dfile)
            for j in labels:
                if len(cache_pos[j]) < cache_size:
                    pos_mask = sample_targets == j
                    if any(pos_mask):
                        cache_pos[j] += list(samples[pos_mask])

        full = []
        for i in labels:
            if len(cache_pos[i]) >= cache_size:
                full.append(True)
            else:
                full.append(False)
        if all(full):
            cache_not_full = False
            print('Positive samples cache full!')
    
    if mode == 'train':

        while True:
            f_idxs = np.random.choice(idxs, size=n)
            features = np.zeros((batch_size, f_W, f_H))
            targets = np.zeros(batch_size)
            for i in range(n):
                file = npy_files[f_idxs[i]]
                dfile = file[0]
                tfile = file[1]
                samples = np.load(dfile)
                sample_targets = np.load(tfile)
                features[i*f_N:(i+1)*f_N] = samples
                targets[i*f_N:(i+1)*f_N] = sample_targets

            # get postive and negative samples
            pos_samples = []
            neg_samples = []
            for i in range(len(targets)):
                targ = int(targets[i])
                idx_pos = np.random.choice(range(len(cache_pos[targ])))
                neg_targets_mask = targets != targets[i]
                neg_targets_masked = targets[neg_targets_mask]
                neg_idx = np.random.choice(np.arange(len(targets[neg_targets_mask])))
                pos_samples.append([cache_pos[targ][idx_pos]])
                neg_samples.append([features[neg_targets_mask][neg_idx]])
                cache_pos[targ][idx_pos] = features[i]
            pos_samples = np.concatenate(pos_samples, axis=0)
            neg_samples = np.concatenate(neg_samples, axis=0)
            
            features = np.concatenate([features, pos_samples, neg_samples], axis=0)
#             targets = np.zeros(features.shape)
            
            if model == 'CNN':
                B, W, H = features.shape
                features = np.reshape(features, (B, W, H, 1))
                
            assert features.shape[0] == 3*batch_size

            yield features
        
    if mode == 'val':
        
        while True:
            
            for j in range(steps):
                features = np.zeros((batch_size, f_W, f_H))
                targets = np.zeros(batch_size)
                for i in range(n):
                    file = npy_files[j+i]
                    dfile = file[0]
                    tfile = file[1]
                    samples = np.load(dfile)
                    sample_targets = np.load(tfile)
                    features[i*f_N:(i+1)*f_N] = samples
                    targets[i*f_N:(i+1)*f_N] = sample_targets

                # get postive and negative samples
                pos_samples = []
                neg_samples = []
                for i in range(len(targets)):
                    targ = int(targets[i])
                    idx_pos = np.random.choice(range(len(cache_pos[targ])))
                    neg_targets_mask = targets != targets[i]
                    neg_targets_masked = targets[neg_targets_mask]
                    neg_idx = np.random.choice(np.arange(len(targets[neg_targets_mask])))
                    pos_samples.append([cache_pos[targ][idx_pos]])
                    neg_samples.append([features[neg_targets_mask][neg_idx]])
                    cache_pos[targ][idx_pos] = features[i]
                pos_samples = np.concatenate(pos_samples, axis=0)
                neg_samples = np.concatenate(neg_samples, axis=0)

                features = np.concatenate([features, pos_samples, neg_samples], axis=0)
#                 targets = np.zeros(features.shape)

                if model == 'CNN':
                    B, W, H = features.shape
                    features = np.reshape(features, (B, W, H, 1))

                assert features.shape[0] == 3*batch_size

                yield features
                
def data_loader_model_wrapper_classification(data_loader, state_shape):
    """
    Interface between data loader and model
    
    args:
        
    returns:
    
    """
    
    while True:
        features, targets = data_loader.__next__()
        N, W, H = features.shape
        W_pad = (W % 16 > 0)*(16 - W % 16)
        H_pad = (H % 16 > 0)*(16 - H % 16)
        features = np.pad(features, ((0,0),(0,W_pad),(0,H_pad)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        N, W, H = features.shape
        input_dict = {
            'x': np.reshape(features, (N, W*H)),
            'logits': targets,
            'state': np.zeros(state_shape)
        }
        yield input_dict
        
def data_loader_model_wrapper_identification(data_loader, state_shape, encoding_dim):
    """
    Interface between data loader and model
    
    args:
        
    returns:
    
    """
    
    while True:
        features = data_loader.__next__()
        N, W, H = features.shape
        W_pad = (W % 16 > 0)*(16 - W % 16)
        H_pad = (H % 16 > 0)*(16 - H % 16)
        features = np.pad(features, ((0,0),(0,W_pad),(0,H_pad)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        N, W, H = features.shape
        input_dict = {
            'x': np.reshape(features, (N, W*H)),
            'state': np.zeros(state_shape),
            'encoding': np.zeros((N, encoding_dim))
        }
        yield input_dict
