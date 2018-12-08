import glob

def get_all_file_names(data_root: str):
    """
    Gets all train and val file names.
    
    args:
        data_root: string holding root directory of data. Needs to contain train and val folder.
        
    returns:
        train_files: training files
        val_files: validation files
    """
    
    # train
    train_target_files = glob.glob(data_root + '/train/**/*_targets.npy', recursive=True)
    train_files = []
    for f in train_target_files:
        spl = f.split('/')
        spl[-1] = spl[-1].split('_')[0] + '.npy'
        data_file = '/'.join(spl)
        train_files.append([data_file, f])

    # val
    val_target_files = glob.glob(data_root + '/val/**/*_targets.npy', recursive=True)
    val_files = []
    for f in val_target_files:
        spl = f.split('/')
        spl[-1] = spl[-1].split('_')[0] + '.npy'
        data_file = '/'.join(spl)
        val_files.append([data_file, f])
        
    return train_files, val_files
