from pycromanager import Dataset
import numpy as np
import time
import matplotlib.pyplot as plt

from skimage import morphology, exposure, filters, measure
from scipy import signal, ndimage

from sdt import roi
from matplotlib.path import Path
import cv2


def open_device_dataset(path, fname_base, device=1, time=0, sufix=''):
    file_name = fname_base + "_A{}_{}h_{}1".format(device, time, sufix)
    data_path = path + "\\" + file_name
    dataset = Dataset(data_path)
    device_meta = {'fname': file_name,
                   'datetime': dataset.summary_metadata['DateAndTime'],
                   'numz': len(dataset.axes['z']),
                   'nump': len(dataset.axes['p']),
                   'numt': len(dataset.axes['time']),
                   'numc': len(dataset.axes['channel']),
                   'pos_list': list(dataset.axes['p'])}

    return dataset, device_meta


def find_chamber(wormchamber, template):
    """opn cv template match. takes in the image and the template,
    returns the x,y pos of the top left of the image
    this implementation """
    res2 = cv2.matchTemplate(wormchamber.astype(np.float32), template.astype(np.float32), method=cv2.TM_CCOEFF);
    (_, _, _, maxLoc) = cv2.minMaxLoc(res2)
    return maxLoc


def find_roi(im, template, verts):
    """takes in the (BF) image, template and a list of x,y tuples as the ROI edge points.
    returns a matplotlib.path

    Calls 'find_chamber'
    depends on opencv as cv2, matplotlib.path as Path

    EX. verts:
    verts = [(0., 0.),  (100., 0.), (180,68) ,(1196., 1068.), (1876,1780) ,(2040., 1992.),  (2048., 2048.),
    (1940., 2040.),  (1732., 1872.),  (1048., 1196.),(90,180) , (0,50),  (0., 0.)]
    """

    loc = find_chamber(im, template)

    for i, v in enumerate(verts):
        x = v[0] + loc[0] - 500
        if x < 0:
            x = 0
        elif x > 2048:
            x = 2048

        y = v[1] - 500 + loc[1]
        if y < 0:
            y = 0
        elif y > 2048:
            y = 2048

        verts[i] = (x, y)
    codes = [Path.MOVETO]
    for i in range(len(verts) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)

    return path, loc

def crop_im(im,path):
    return roi.PathROI(path)(im)


def measure_sharpness(im,mediansize = 3,filt = None, pool = True):
    if pool:
        im = measure.block_reduce(im,(mediansize,mediansize),func = np.median)
    else:
        im = ndimage.median_filter(im,mediansize)
    if filt is None:
        filt = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    im = signal.convolve2d(im,filt,mode = 'same')
    return np.sum(im**2)

def z_stack_sharpness(dataset,p,numz,t = 1,c = 1,pool = True):
    sharpness = []
    for z in range(numz):
        im = dataset.read_image(channel = c,z = z,time = t,p = p)
        sharpness.append(measure_sharpness(im,pool = pool))
    return sharpness


def measure_sharpness_diag(im,mediansize = 3,filt = None, pool = True):
    if pool:
        im = measure.block_reduce(im,(mediansize,mediansize),func = np.median)
    else:
        im = ndimage.median_filter(im,mediansize)
    if filt is None:
        filt = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    im = signal.convolve2d(im,filt,mode = 'same')
    s = 0
    rcut = int(im.shape[0]/4)
    for i in range(im.shape[0]):
        for j in range(im.shape[0]):
            if i>j-rcut and i<j+rcut:
                s+=im[i,j]**2
    return s

def z_stack_sharpness_diag(dataset,p,numz,t = 1,c = 1,pool = True,norm = True):
    sharpness = []
    for z in range(numz):
        im = dataset.read_image(channel = c,z = z,time = t,p = p)
        sharpness.append(measure_sharpness_diag(im,pool = pool))
    if norm:
 #         norm = np.linalg.norm(sharpness)
 #         normal_array = sharpness/norm
        return sharpness/np.sum(sharpness)
    else:
        return sharpness

def whole_device_sharpness(dataset,device_meta,t=1,c=1,save_meta = True):
    pos_sharpness = []
    for p in device_meta['pos_list']:
        pos_sharpness.append(z_stack_sharpness_diag(dataset, p=p, numz=10, t=t, c=c))
    if save_meta:
        device_meta['sharpness'] = pos_sharpness
    return pos_sharpness

def create_diagonal_zstrip(dataset,crop_path,delta, worm_n,p,c = 1,t=1):
    length = 2048 + delta * (worm_n - 1)
    strip = np.zeros((2048, length))

    c = 1
    t = 1
    #sum_projection = np.zeros((crop_im.shape[0], crop_im.shape[1]))
    for i in range(worm_n):
        image = dataset.read_image(channel=c, z=i, time=t, p=p)
        #maxp = np.max(image)
        #minp = np.min(image)
        crop = roi.PathROI(crop_path)(image)
        #
        # if i > 1 and i < 9:
        #     sum_projection += crop

        rows = crop.shape[0]
        cols = crop.shape[1]
        strip[:rows, i * delta:i * delta + cols] += crop
    return strip

def sum_projection(dataset,start_z,end_z):
    sum_projection = np.zeros((crop_im.shape[0], crop_im.shape[1]))
    for z in range(start_z,end_z)
