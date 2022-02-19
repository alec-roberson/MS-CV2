import argparse

if __name__ == '__main__':
    DATA_AUG = dict(
        mosaics = 0.25,
        mixup = 0.25,
        cutmix = 0.25,
        cutout = 0.25,
        hflip = 0.25,
        vflip = 0.25,
        rot = 0.25)
    
    from datamanager import DataManager, DataLoader

    # dl = DataLoader('premade-batches')
    # dm = DataManager('data/test-416.pt', batch_size=32, mini_batch_size=16, CUDA=True, **DATA_AUG)
    dm = DataManager('data/train-416.pt', batch_size=32, mini_batch_size=16, CUDA=True, **DATA_AUG)

