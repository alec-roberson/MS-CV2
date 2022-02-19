import argparse

if __name__ == '__main__':
    DATA_AUG = dict(
        mosaics = 0.30,
        mixup = 0.30,
        cutmix = 0.30,
        cutout = 0.0,
        hflip = 0.25,
        vflip = 0.05,
        rot = 0.05)
    
    from datamanager import DataManager

    dm = DataManager('data/train-416.pt', 32, 16, True, **DATA_AUG)

    dm.premake_batches('pmb-200', 200)
    