''' train.py
this file trains the network.
'''

# +++ imports
import json
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from network.model import NetworkModel
from datamanager import DataManager

if __name__ == '__main__':
    # +++ setup the parser
    parser = argparse.ArgumentParser()
    # 
    # network
    network_args = parser.add_argument_group('network args')
    network_args.add_argument('--net-config', '-nc',
        type=str, default=None, dest='net_config')
    network_args.add_argument('--net-base-name', '-nbn',
        type=str, default=None, dest='net_base_name')
    # data
    data_args = parser.add_argument_group('data args')
    data_args.add_argument('--train-data', '-trd',
        type=str, default=None, dest='train_data')
    data_args.add_argument('--test-data', '-ted',
        type=str, default=None, dest='test_data')
    # data aug
    data_aug_args = parser.add_argument_group('data augmentation args')
    data_aug_args.add_argument('--mosaic',
        type=float, default=None, dest='mosaic')
    data_aug_args.add_argument('--mixup',
        type=float, default=None, dest='mixup')
    data_aug_args.add_argument('--cutmix',
        type=float, default=None, dest='cutmix')
    data_aug_args.add_argument('--cutout',
        type=float, default=None, dest='cutout')
    data_aug_args.add_argument('--hflip',
        type=float, default=None, dest='hflip')
    data_aug_args.add_argument('--vflip',
        type=float, default=None, dest='vflip')
    data_aug_args.add_argument('--rot',
        type=float, default=None, dest='rot')
    data_aug_args.add_argument('--iou_thresh', nargs=2,
        type=float, default=None, dest='iou_thresh')
    data_aug_args.add_argument('--cut_range', nargs=2,
        type=float, default=None, dest='cut_range')
    # training parameters
    training_args = parser.add_argument_group('training args')
    training_args.add_argument('--batch-size', '-bs',
        type=int, default=None, dest='batch_size')
    training_args.add_argument('--mini-batch-size', '-mbs',
        type=int, default=None, dest='mini_batch_size')
    training_args.add_argument('--learning-rate', '-lr',
        type=float, default=None, dest='learning_rate')
    training_args.add_argument('--weight-decay', '-wd',
        type=float, default=None, dest='weight_decay')
    training_args.add_argument('--momentum', '-mom',
        type=float, default=None, dest='momentum')
    # tensorboard args
    tb_args = parser.add_argument_group('tensorboard args')
    tb_args.add_argument('--write-every',
        type=int, default=None, dest='write_every')
    tb_args.add_argument('--tb-logs-dir',
        type=str, default=None, dest='tb_logs_dir')
    # cuda args
    cuda_args = parser.add_argument_group('cuda args')
    cuda_args.add_argument('--cuda-device',
        type=str, default=None, dest='cuda_device')
    cuda_args.add_argument('--aug-cuda',
        type=bool, default=None, dest='augmentation_cuda')
    cuda_args.add_argument('--train-cuda',
        type=bool, default=None, dest='training_cuda')
    # +++ read json and parse the arguments
    
        
    args = parser.parse_args()
    print(args.iou_thresh, args.cut_range)
    quit()



# +++ file locations
NET_CFG = 'cfg/yolov3-mid-256.cfg' # 'cfg/squeezedet.cfg' # network configuration file
TRAIN_FILE = 'T-test-256.pt' # training images/labels directory
TEST_FILE = 'T-test-256.pt' # testing images/labels directory
NET_NAME = 'TESTING2-yv3m-256' # where to save the network after training
TB_LOGDIR = 'runs/' + NET_NAME # log for tensorboard
SAVE_NET = NET_NAME + '.pt' # file to save net to

# +++ data preferences
BATCH_SIZE = 10 # batch size
MINI_BATCH_SIZE = 10 # mini batch size
AUG_CUDA = False # should data be used for data augmentation
DATA_AUG = {}
#     'mosaics': 0.25,
#     'mixup': 0.25,
#     'cutmix': 0.25,
#     'cutout': 0.25,
#     'hflip': 0.25,
#     'vflip': 0.25,
#     'rot': 0.25
# }

# +++ training prefrences
CUDA = False # should cuda be used for the network
NUM_EPOCHS = 2 # number of epochs to run for
WRITE_EVERY = 1 # tensorboard data will be written every ___ epochs

# +++ optimizer prefrences
LEARNING_RATE = 0.001 # learning rate
WEIGHT_DECAY = 0.001 # learning rate decay
MOMENTUM = 0.9 # momentum for SGD optimizer
NESTEROV = False # nesterov SGD?

# +++ set the device variable
device = 'cuda:0' if CUDA else 'cpu'

# +++ setup the network
model = NetworkModel(NET_CFG, CUDA=CUDA)

model.to(device) # send network to the right device
model.train() # put in training mode

# +++ load up them datas
train_data = DataManager(
    data_path = TRAIN_FILE,
    CUDA=AUG_CUDA,
    **DATA_AUG)
test_data = DataManager(
    data_path = TEST_FILE,
    CUDA=AUG_CUDA)

# +++ setup the optimizer and shit
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=NESTEROV)

# +++ tensorboard setup 
writer = SummaryWriter(log_dir=TB_LOGDIR)

# +++ test network function
def test_model():
    # get the data and targets to test on
    data, targets = test_data.batches()[0]
    data, targets = data.to(device), targets.to(device)
    
    # test the network
    model.eval()
    with torch.no_grad():
        _, loss = model(data, targets=targets)
    model.train()
    
    # return the loss
    return loss.item()

# +++ log metrics function
def log_metrics(step):
    metrics = model.metrics # get the metrics
    # +++ first, plotting the metrics collected by the model
    plots = [
        ('loss', 'loss'),
        ('loss breakdown', ('bbox-loss', 'obj-conf-loss', 'noobj-conf-loss', 'cls-loss')),
        ('confidence', ('conf-obj', 'conf-noobj')),
        ('class accuracy', 'cls-accuracy'),
        ('percent correct detections', ('recall50', 'recall75')),
        ('precision', 'precision')]
    
    for plot_name, plot_keys in plots:
        if isinstance(plot_keys, str): 
            # single value plot
            writer.add_scalar(plot_name, metrics[plot_keys], global_step=step)
        else:
            # multivalue plot
            plot_dict = dict([(k, metrics[k]) for k in plot_keys])
            writer.add_scalars(plot_name, plot_dict, global_step=step)
    
    # +++ make the class accuracy by class
    cls_acc_bd = list(metrics['cls-acc-by-cls'])
    plot_data = dict(zip(train_data.classes, cls_acc_bd))
    writer.add_scalars('class accuracy by class', plot_data, global_step=step)

    loss = test_model()

    writer.add_scalar('test loss', loss, global_step=step) # add it to tensorboard

    # +++ and lastly, reset the model's metrics
    model.reset_metrics() # reset the model's metrics

# +++ main training loop
for epoch in tqdm(range(1, NUM_EPOCHS+1), 'training'):

    # +++ loop through batches in this epoch
    for batch in train_data.batches(batch_size=BATCH_SIZE, mini_batch_size=MINI_BATCH_SIZE):
        model.zero_grad() # zero the model's gradients

        batch_loss = 0. # batch loss

        for x, y in batch:
            x = x.to(device) # send x to right device
            y = y.to(device) # send targets to the right device

            _, mini_batch_loss = model(x, targets=y) # feed through the network
            batch_loss += mini_batch_loss
        
        batch_loss.backward() # backpropogate all batch loss
        optimizer.step() # take a step with the optimizer

    # +++ check if we should write data now!
    if epoch % WRITE_EVERY == 0:
        log_metrics(epoch) # log the metrics for this epoch

    # +++ update the dropblocks
    model.set_db_kp(epoch / (NUM_EPOCHS - 1))

# +++ lastly, test the model and print out the final metrics
model.reset_metrics()
test_model()
metrics = model.metrics

print('training complete. final test metrics are:')
model.print_metrics()

model.reset_metrics()
model.eval()

# +++ save the model
f = open(SAVE_NET, 'wb')
torch.save(model, f)
f.close()
