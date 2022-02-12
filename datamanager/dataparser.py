''' dataparser.py
this file contains the DataParser class, along with some functions that are
useful for loading data from files.
'''
# +++ IMPORTS
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from datamanager.datamanager import DataManager

# # set the backend
# matplotlib.use("Qt5agg")

# +++ FUNCTIONS
def load_classes(class_file):
    ''' load classes function
    loads classes from a .txt file into a list.

    --- args ---
    path : str
        path to the file containing the class names.

    --- returns ---
    list[str] : list of class names in order, as read from the file.
    '''
    f = open(class_file, 'r')
    names = f.read().split('\n')
    f.close()
    names = [name.strip() for name in names] # strip spaces
    names = [name for name in names if name != ''] # ignore empty lines
    return names

def get_matching_files(data_path, suppress_warnings=False):
    ''' get matching files function
    this function looks through a directory and gets the names (without
    extensions) of all the [name].jpg files that also have a corresponding
    [name].txt file.

    --- args ---
    data_path : str
        the *real* path to the directory to get the file names from.
    suppress_warnings : bool, optional (default=False)
        if true, no warnings will be printed for unmatched data.

    --- returns ---
    list[str] : the list of filenames with .jpg and .txt files in the data_path.
    '''
    # read all files in the path and get rid of sub-directories
    all_files = os.listdir(data_path)
    all_files = list(filter(lambda f: '.' in f, all_files))
    # loop through the files to generate the output
    out = []
    while len(all_files) > 0:
        f = all_files.pop(0)
        # get the extension and name
        ext = f.split('.')[-1]
        fname = f[:- len(ext) - 1]
        if ext == 'jpg' and fname + '.txt' in all_files:
            out.append(fname)
            all_files.remove(fname + '.txt')
        elif ext == 'txt' and fname + '.jpg' in all_files:
            out.append(fname)
            all_files.remove(fname + '.jpg')
        elif suppress_warnings == False:
            print(f'warning: file {f} in {data_path} has no matching image/label')
    return out

def load_labels(pdir, names):
    ''' load labels function
    reads specified label files from a directory and returns a numpy array of
    the label tensors in the order of file names given.

    --- args ---
    pdir : str
        the parent directory that the label files are in.
    names : list[str]
        the NAMES (no extension!) of the label files to read, in order.

    --- returns ---
    np.ndarray[torch.FloatTensor] : a numpy array containing the label tensors,
        with shape (_, 1).
    '''
    # put together path and names to get the file names
    fnames = [os.path.join(pdir, f'{name}.txt') for name in names]
    labels = np.empty((len(fnames),1), dtype=object)
    for i, fname in enumerate(fnames):
        f = open(fname, 'r')
        lbl = f.read().split('\n')
        f.close()
        lbl = [l.strip() for l in lbl]
        lbl = [l for l in lbl if l != '']
        # read each line as a list of numbers
        lbl = [ [float(n) for n in line.strip().split(' ') if n != '']
                for line in lbl]
        lbl = torch.FloatTensor(lbl).reshape(-1, 5)
        labels[i,0] = lbl
    return labels

def load_images(pdir, names, resize_dim):
    ''' load images function
    loads images from a specified directory and returns a numpy array containing
    all the data about the images.

    --- args ---
    pdir : str
        the parent directory the image files are in.
    names : list[str]
        the names (no extension!) of the image files to load from the directory.
    resize_dim : int
        the square size images should be resized to.
    
    --- returns ---
    np.ndarray[object] : the data array with size (num_images, 4), where the 4
        values in the first dimension are: data point name, original image,
        original dimensions, and resized image.
    '''
    # turn the file names into image files
    fnames = [os.path.join(pdir, f'{n}.jpg') for n in names]
    # loop through the files to make the output
    out = np.empty((len(fnames), 4), dtype=object)
    for i, (imgf, name) in enumerate(zip(fnames, names)):
        # get the original image
        orig_img = cv2.imread(imgf)
        orig_dims = torch.FloatTensor(orig_img.shape[:2]).flip(0)
        # get the new image
        img = cv2.resize(orig_img, (resize_dim, resize_dim))
        img = torch.FloatTensor(img).div(255).permute(2,0,1).flip(0)
        # write output
        out[i,0] = name
        out[i,1] = orig_img
        out[i,2] = orig_dims
        out[i,3] = img
    return out

# +++ CLASS : DataParser
class DataParser:
    ''' data parser
    this object is used to load, save, split, or otherwise manipulate data sets.
    this is very useful and saves time when loading large data sets.

    --- args ---
    raw_data_path : string
        the path to the folder containing the raw data to load. this should be
        in the form of image files ([filename].jpg) with corresponding labels
        ([filename].txt) in the YOLO format.
    class_file : string
        the file containing the classes of the data set with one on each line.
    resize_dim : int
        the square dimension that images should be resized to for the data set.
    suppress_warnings : bool, optional (default=False)
        if true, you will not be warned about unmatching data in the directory
        provided.
    '''
    def __init__(self, raw_data_path, class_file, resize_dim, suppress_warnings=False):
        self.raw_data_path = raw_data_path
        # load the class names
        self.classes = load_classes(class_file)
        # load all the data
        fnames = get_matching_files(raw_data_path,
            suppress_warnings=suppress_warnings)
        self.data = np.concatenate((
            load_images(raw_data_path, fnames, resize_dim),
            load_labels(raw_data_path, fnames)), axis=1)
        self.resize_dim = resize_dim

    def save(self, default_fn='data', *fns_and_pcts, shuffle=True):
        ''' save data method
        splits the data into a training and testing set, and then saves both to
        binary .pt files.

        --- args ---
        default_fn : str, optional (default='data')
            The default filename (no extension) to save the data to.
        *fns_and_pcts
            Other files that data should be allocated to. Each filename should
            be immediately followed by a float between 0 and 1 indicating the
            percentage of data to allocate to said file.
        shuffle : bool, optional (defualt=True)
            should the data be shuffled before being saved.
        
        --- returns ---
        int : number of images in training set
        int : number of images in testing set.
        '''
        # make a copy of the data
        data = self.data.copy()
        # shuffle it
        if shuffle: np.random.shuffle(data)
        # unpack the input filenames
        fns_and_pcts = list(fns_and_pcts)
        n_aux_files = int(len(fns_and_pcts)/2)
        fns = []
        pcts = []
        if len(fns_and_pcts) % 2 != 0:
            raise ValueError('DataParser.save was not given an even number of arguments. Each file must be followed with a percentage of data to allocate to that file.')
        else:
            for i in range(0, len(fns_and_pcts), 2):
                fns.append(fns_and_pcts[i])
                pcts.append(fns_and_pcts[i+1])
            if sum(pcts) > 1:
                raise ValueError('DataParser.save was given a total percentage of data to save >100%')
        # get indexes to portion out the data
        N = data.shape[0]
        idxs = [0] + [int(p*N) for p in pcts]
        # loop to split up and save the data
        out = []
        for i in range(n_aux_files):
            num = idxs[i+1] - idxs[i] - 1
            out.append((f'{fns[i]}.pt', num))
            torch.save((data[idxs[i]:idxs[i+1]], self.classes, self.resize_dim), f'{fns[i]}.pt')
        torch.save((data[idxs[-1]:], self.classes, self.resize_dim), f'{default_fn}.pt')
        out.append((f'{default_fn}.pt', N - idxs[-1]))
        return out
    
    def save_raw(self, output_dir:str, keep_fns=False, round_lbls_to=6) -> None:
        ''' save_raw
        Saves the raw image files to the directory given.

        Parameters
        ----------
        output_dir : str
            The output directory that the new files should be saved to.
        '''
        # setup the file names
        if keep_fns:
            filenames = list(self.data[:,0])
        else:
            n = self.data.shape[0]
            ndig = int(np.ceil(np.log10(n)))
            filenames = [f'img{i:0>{ndig}}' for i in range(n)]
        # configure the output folder
        if os.path.exists(output_dir):
            raise FileExistsError(f'output directory \'{output_dir}\' already exists! process aborted')
        else:
            os.makedirs(output_dir)
        # save all the things
        for i, fn in enumerate(filenames):
            # save the image
            cv2.imwrite(
                os.path.join(
                    output_dir, f'{filenames[i]}.jpg'), 
                self.data[i][1])
            # put together and save the label
            lbl_file_txt = ''
            for lbl in self.data[i][4]:
                lbl_file_txt += str(int(lbl[0])) + ' '
                lbl_file_txt += ' '.join([f'{x.item():.{round_lbls_to}}' for x in lbl[1:]]) + '\n'
            # save the label
            with open(os.path.join(
                output_dir, f'{filenames[i]}.txt'), 'w+') as f:
                f.write(lbl_file_txt)

    def trim_data(self, temp_data_fn='~TEMPDATA.pt'):
        ''' trim data
        this function allows for the user to comb through each data point and
        decide to keep it or not.

        --- args ---
        temp_data_fn : str, optional (default='~TEMPDATA.pt')
            the file where the data will temporarily be stored.
        '''
        # temporarily save the data
        torch.save((self.data, self.classes, self.resize_dim), temp_data_fn)
        # load the data into a datamanager
        dm = DataManager(temp_data_fn)
        # draw boxes on the images
        det_imgs = dm.write_boxes(dm.get_lbls())
        # delete temporary file
        os.remove(temp_data_fn)
        # setup the plot
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        plt.show(block=False)
        # loop to let user select images
        keep_idxs = []
        for i in range(self.data.shape[0]):
            # put the image onto the plot
            img = np.flip(det_imgs[i], axis=2)
            ax.set_title(f'img # {i}')
            ax.imshow(img)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            # get the input and record the index
            inp = input(f'{i} > ')
            if inp == '':
                keep_idxs.append(i)
            elif inp.lower() == 'quit':
                quit()
            # clear the axes
            ax.cla()
        # close the plot
        plt.close()
        # only keep the specified images
        self.data = self.data[keep_idxs]
