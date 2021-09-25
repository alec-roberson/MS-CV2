''' __main__.py
This is the main script to be executed for the dataconfig module.
'''
import argparse
from datamanager import DataParser

# +++ FUNCTIONS
def parse_data(args):
    ''' parse_data function
    Performs the parse_data action for the dataconfig package.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed for data parsing.
    '''
    dp = DataParser(args.data_path, args.class_names_file, args.resize_dim)
    dp.save(args.train_out, args.test_out, test_pct=args.pct_test, shuffle=args.shuffle)

# +++ MAIN SCRIPT
if __name__ == '__main__':
    # +++ imports
    import argparse

    # +++ argument parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='function', dest='function')
    
    # regular data parsing
    pd_parser = subparsers.add_parser('pd', help='parse and split data')
    pd_parser.add_argument('data-path', dest='data_path', type=str, 
        help='the path to the folder containing the raw data information')
    pd_parser.add_argument('class-names-file', dest='class_names_file', type=str,
        help='the path to the file containing the ordered class names')
    pd_parser.add_argument('resize-dim', dest='resize_dim', type=int,
        help='the size that each image will be resized to in the resulting dataset')
    pd_parser.add_argument('--train-out', dest='train_out', type=str, default='train-data.pt',
        help='the path to the output file for the training data (no extension!)')
    pd_parser.add_argument('--test-out', dest='test_out', type=str, default='test-data.pt',
        help='the path to the output file for the testing data (no extension!)')
    pd_parser.add_argument('--pct-test', dest='pct_test', type=float, default=0.1,
        help='the percentage of data that should be allocated to testing')
    pd_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
        help='if added, the data will not be shuffled')
    
    args = parser.parse_args()

    if args.function == 'pd':
        parse_data(args)
