import argparse

if __name__ == '__main__':
    # make a parser and the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', type=str, help='config file for the network')
    parser.add_argument('-name', '--net-name', type=str, help='name the network is given', dest='net_name', default=None)
    parser.add_argument('-n', '--num-epochs', type=int, help='number of epochs to train for', default=32, dest='num_epochs')
    # get the stuff
    args = parser.parse_args()
    
    
    print(f'config file: \'{args.cfg_file}\'')
    print(f'network name: \'{args.net_name}\'')
    if args.net_name:
        # 
    print(f'number of epochs: \'{args.num_epochs}\'')