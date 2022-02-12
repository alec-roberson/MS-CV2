''' __main__.py
This is the main script to be executed for the dataconfig module.
'''
import os
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

    aux_args = []
    if args.aux_out:
        for i in range(len(args.aux_out)):
            a = args.aux_out[i]
            if i % 2 == 1:
                aux_args.append(float(a))
            else:
                aux_args.append(str(a))
    
    dp = DataParser(args.data_path, args.class_names_file, args.resize_dim)
    out = dp.save(args.default_fn, *aux_args, shuffle=args.shuffle)
    for f, n in out:
        print(f'file \'{f}\' contains {n} data points')

def trim_data(args):
    ''' trim_data function
    Performs the functionality of trimming data.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed for data parsing.
    '''
    if os.path.exists(args.output_data_path):
        raise FileExistsError(f'output directory \'{args.output_data_path}\' already exists! action aborted')
    dp = DataParser(args.data_path, args.class_names_file, 16)
    dp.trim_data()
    dp.save_raw(args.output_data_path,
        keep_fns = args.keep_fns)

def lump_classes(args):
    raise NotImplementedError()

# +++ MAIN SCRIPT
if __name__ == '__main__':
    # +++ imports
    import argparse

    # +++ argument parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='function', dest='function')
    
    # regular data parsing
    pd_parser = subparsers.add_parser('pd', help='parse and split data')
    pd_parser.add_argument('data_path',type=str,
        help='the path to the folder containing the raw data files')
    pd_parser.add_argument('class_names_file', type=str,
        help='the path to the file containing the ordered class names')
    pd_parser.add_argument('resize_dim', type=int,
        help='the size that each image will be resized to in the resulting dataset')
    pd_parser.add_argument('-o', '--out', dest='default_fn', type=str, default='data',
        help='the default file to save data to (defualt=data)')
    pd_parser.add_argument('-a', '--aux-out', dest='aux_out', type=str, default=None,
        nargs='*',
        help='auxillary data output, list file names immediately followed by '
        'the percentage of data between 0 and 1 that should be allocated to the file.')
    pd_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
        help='if added, the data will not be shuffled')

        

    # trim data parser
    td_parser = subparsers.add_parser('td', help='trim out bad data from the dataset')
    td_parser.add_argument('data_path', type=str,
        help='the path to the folder containing the raw data files')
    td_parser.add_argument('class_names_file', type=str,
        help='the path to the file containing the ordered class names')
    td_parser.add_argument('output_data_path', type=str,
        help='the path to a new directory where the output data should be saved.')
    td_parser.add_argument('--keep_filenames', '-kfn', dest='keep_fns', action='store_true',
        help='if triggered, the filenames will be kept from the original directory.')

    # lump classes parser
    lc_parser = subparsers.add_parser('lc', help='lump classes together')
    lc_parser.add_argument('new_classes', type=str, help='the path to the file containing the new set of classes.')
    lc_parser.add_argument('class_mapping', type=str, help='the json file containing the mapping of class numbers (old -> new/str -> int).')
    lc_parser.add_argument('data_files', type=str, nargs='+', help='the path(s) to the data file(s) that you wish to remap the classes for. you must supply an even number of arguments with each file being followed by the file you want the new data to be saved to.')
    

    args = parser.parse_args()

    if args.function == 'pd':
        parse_data(args)
    elif args.function == 'td':
        trim_data(args)
