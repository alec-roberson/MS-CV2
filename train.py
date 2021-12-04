''' train.py
this file trains the network.
'''

# +++ imports
from tqdm import tqdm
import torch 
import torch.optim as optim
import argparse
import configparser
from torch.utils.tensorboard import SummaryWriter

from network.model import NetworkModel
from datamanager import DataManager

# +++ set to None
NET_CFG, TRAIN_FILE, TEST_FILE, NET_NAME, BATCH_SIZE, MINI_BATCH_SIZE, AUG_CUDA, DATA_AUG, CUDA, NUM_EPOCHS, WRITE_EVERY, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM = None, None, None, None, None, None, None, None, None, None, None, None, None, None

# +++ new args parse

if __name__ == '__main__':
    # make a parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-train_cfg', type=str, help='config file for training', dest='TRAIN_CFG', default='train.cfg')

    # add file locations
    parser.add_argument('-net_cfg', type=str, help='config file for the network', dest='NET_CFG', default=None)
    parser.add_argument('-train', type=str, help='training images/labels directory', dest='TRAIN_FILE', default=None)
    parser.add_argument('-test', type=str, help='testing images/labels directory', dest='TEST_FILE', default=None)
    parser.add_argument('-name', type=str, help='where to save the network after training', dest='NET_NAME', default=None)
    
    # add gpu preferences
    parser.add_argument('-cuda', type=bool, help='should cuda be used for the network', dest='CUDA', default=None)
    parser.add_argument('-aug_cuda', type=bool, help='should data be used for data augmentation', dest='AUG_CUDA', default=None)
    
    # add data preferences
    parser.add_argument('-batch', type=int, help='batch size', dest='BATCH_SIZE', default=None)
    parser.add_argument('-mini_batch', type=int, help='mini batch size', dest='MINI_BATCH_SIZE', default=None)

    # add training preferences
    parser.add_argument('-epochs', type=int, help='number of epochs to run for', dest='NUM_EPOCHS', default=None)
    parser.add_argument('-write', type=int, help='how often data will be written to the tensorboard in epochs', dest='WRITE_EVERY', default=None)

    # add optimizer preferences
    parser.add_argument('-learning', type=float, help='learning rate', dest='LEARNING_RATE', default=None)
    parser.add_argument('-weight', type=float, help='learning rate decay', dest='WEIGHT_DECAY', default=None)
    parser.add_argument('-momentum', type=float, help='momentum for SGD optimizer', dest='MOMENTUM', default=None)

    # add data augmentation preferences
    parser.add_argument("-da_mosaics", type=float, help='mosaic data augmentation', dest='DA_MOSAICS', default=None)
    parser.add_argument("-da_mixup", type=float, help='mixup data augmentation', dest='DA_MIXUP', default=None)
    parser.add_argument("-da_cutmix", type=float, help='cut mix data augmentation', dest='DA_CUTMIX', default=None)
    parser.add_argument("-da_cutout", type=float, help='cut out data augmentation', dest='DA_CUTOUT', default=None)
    parser.add_argument("-da_hflip", type=float, help='horizontal flip data augmentation', dest='DA_HFLIP', default=None)
    parser.add_argument("-da_vflip", type=float, help='vertical flip data augmentation', dest='DA_VFLIP', default=None)
    parser.add_argument("-da_rot", type=float, help='rotation data augmentation', dest='DA_ROT', default=None)

    # get the stuff
    args = parser.parse_args()

    config = configparser.ConfigParser(inline_comment_prefixes=['#'])
    config.read(args.TRAIN_CFG)

    prefs = list(config['network-config'].keys())
    for p in prefs:
        arg = args.__dict__[p.upper()]
        try:
            default = eval(config['network-config'][p])
        except:
            default = config['network-config'][p]

        globals()[p.upper()] = args.__dict__[p.upper()] if args.__dict__[p.upper()] else default

    # +++ initializes DATA_AUG dictionary
    DATA_AUG = dict([(k.lower()[3:],v) for k,v in globals().items() if k.startswith('DA_')])
    
    # +++ file locations
    TB_LOGDIR = 'runs/' + NET_NAME # log for tensorboard
    SAVE_NET = NET_NAME + '.pt' # file to save net to

    # +++ set the device variable
    device = 'cuda:0' if CUDA else 'cpu'

    # +++ setup the network
    model = NetworkModel(NET_CFG, CUDA=CUDA)

    model.to(device) # send network to the right device
    model.train() # put in training mode

    # +++ load up them datas
    train_data = DataManager(
        data_path = TRAIN_FILE,
        batch_size=BATCH_SIZE, 
        mini_batch_size=MINI_BATCH_SIZE,
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
        weight_decay=WEIGHT_DECAY)

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
        for batch in train_data.batches():
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
    torch.save(model.state_dict(), f)
    f.close()
