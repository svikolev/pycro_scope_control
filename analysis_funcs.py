from pycromanager import Dataset
import numpy as np
import time
import matplotlib.pyplot as plt

from skimage import morphology, exposure, filters, measure
from scipy import signal, ndimage

from sdt import roi
from matplotlib.path import Path
import cv2
import json
import os
import tifffile


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
    verts_adjust = []

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
        verts_adjust.append((x, y))
        #verts[i] = (x, y)
    codes = [Path.MOVETO]
    for i in range(len(verts_adjust) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(verts_adjust, codes)

    return path, loc

def find_roi_from_loc(verts,loc):
    verts_adjust = []

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
        verts_adjust.append((x, y))
        # verts[i] = (x, y)
    codes = [Path.MOVETO]
    for i in range(len(verts_adjust) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(verts_adjust, codes)

    return path, loc

def crop_im(im,path,**kwargs):
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

def whole_device_sharpness(dataset,device_meta,t=1,c=1,save_meta = False):
    pos_sharpness = []
    for p in device_meta['pos_list']:
        pos_sharpness.append(z_stack_sharpness_diag(dataset, p=p, numz=10, t=t, c=c).tolist())
    if save_meta:
        device_meta['sharpness'] = pos_sharpness
    return pos_sharpness


def find_loc_all_worms(dataset,device_meta,template,t=0,c=0,save_meta = True):
    loc_list = []
    z = 0
    for p in device_meta['pos_list']:
        im = dataset.read_image(channel=c, z=z, time=t, p=p)
        loc = find_chamber(im,template)
        loc_list.append(loc)
    if save_meta:
        device_meta['loc_list'] = loc_list
    return loc_list



# def find_loc_path_all_worms(dataset,device_meta,template,verts,t=0,c=0,save_loc = True,save_path = False):
#     loc_list = []
#     path_list = []
#     z = 0
#     for p in range(device_meta['nump']):
#         im = dataset.read_image(channel=c, z=z, time=t, p=p)
#         loc = find_chamber(im,template)
#         loc_list.append(loc)
#         path,_ = find_roi_from_loc(verts,loc)
#         path_list.append()
#     if save_loc:
#         device_meta['loc_list'] = loc_list
#     i
#
#     return loc_list,

def get_edof_files(path,timep,device,worm):
    ## there is typo in the naming of files. file is ouput instead of output

    hmap_file_name = os.path.join(path,"hmap_{}timep_{}dev_{}worm.tif".format(timep,device,worm))
    output_file_name = os.path.join(path, "ouput_{}timep_{}dev_{}worm.tif".format(timep, device, worm))
    hmap = tifffile.imread(os.path.join(path, hmap_file_name))
    output = tifffile.imread(os.path.join(path, output_file_name))
    return output,hmap

def create_diagonal_zstrip(dataset,crop_path,delta, worm_n,p,c = 1,t=1):
    length = 2048 + delta * (worm_n - 1)
    strip = np.zeros((2048, length))
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

def sum_projection(dataset,start_z,end_z,p,c = 1,t = 1,crop_path = None):
    # bounds = crop_path.get_extents().bounds
    # num_rows = int(bounds[3] - bounds[1])
    # num_cols = int(bounds[2] - bounds[0])
    """ start_z is included, end_z is not included"""
    sum_projection = None #np.zeros((num_rows,num_cols))
    for z in range(start_z,end_z):
        if sum_projection is not None:
            sum_projection += dataset.read_image(channel=c, z=z, time=t, p=p)
        else:
            sum_projection = np.array(dataset.read_image(channel=c, z=z, time=t, p=p))
    if crop_path is not None:
        return crop_im(sum_projection,crop_path)
    else:
        return sum_projection

def max_projection():
    return None

def find_roi_sum_proj(dataset,center,verts,start_z,end_z,p):
    BF = np.array(dataset.read_image(channel=0, z=0, time=0, p=p))
    crop_path, _ = find_roi(BF, center, verts)
    BF_crop = crop_im(BF,crop_path)
    s_proj = sum_projection(dataset, start_z, end_z, p, c=1, t=1, crop_path=crop_path)
    return BF_crop,s_proj

def find_roi_sum_proj_all_worms(dataset,device_meta,center,verts,start_z,end_z,keep_BF = False):
    num_worms = len(device_meta['pos_list'])
    pos_stack_RL = np.zeros((num_worms,2048,2048))
    if keep_BF:
        pos_stack_BF = np.zeros((num_worms, 2048, 2048))
    else:
        pos_stack_BF = None
    for i,p in enumerate(device_meta['pos_list']):
        BF = dataset.read_image(channel=0, z=0, time=0, p=p)
        crop_path, _ = find_roi(BF, center, verts)
        s_proj = sum_projection(dataset, start_z, end_z, p, c=1, t=1, crop_path=crop_path)
        num_rows = s_proj.shape[0]
        num_cols = s_proj.shape[1]
        pos_stack_RL[i,0:num_rows,0:num_cols] = s_proj
        if keep_BF:
            BF_crop = crop_im(BF, crop_path)
            pos_stack_BF[i, 0:num_rows, 0:num_cols] = BF_crop
    return pos_stack_RL,pos_stack_BF


def create_diagonal_posstrip(stack,delta):
    worm_n = stack.shape[0]
    length = 2048 + delta * (worm_n - 1)
    strip = np.zeros((2048, length))
    #sum_projection = np.zeros((crop_im.shape[0], crop_im.shape[1]))
    for i in range(worm_n):
        image = np.squeeze(stack[i,:,:])
        rows = image.shape[0]
        cols = image.shape[1]
        strip[:rows, i * delta:i * delta + cols] += image
    return strip


def save_diagonal_strip():
    return None

def compute_back_sub_tot_F(RL_stack,device_meta,thresh):
    tot_F_sum_proj = []
    numw = RL_stack.shape[0]
    for w in range(numw):
        tot_F_sum_proj.append(np.sum(np.fmax(np.squeeze(RL_stack[w, :, :]) - thresh, 0)))
    device_meta['tot_F_sum_proj'] = tot_F_sum_proj
    return tot_F_sum_proj


def compute_non_zero_otsu(strip):
    flat_proj = strip.flatten()
    non_zero = flat_proj[flat_proj != 0]
    otsu_thresh = filters.threshold_otsu(non_zero)
    return otsu_thresh

def save_device_meta_json(device_meta,path,fname):
    j = json.dumps(device_meta)
    f = open(os.path.join(path,fname), "w")
    f.write(j)
    f.close()

def open_device_meta_json(path):
    f = open(path, "r")
    data = json.loads(f.read())
    f.close()
    return data

def open_json(path):
    f = open(path, "r")
    data = json.loads(f.read())
    f.close()
    return data

def save_json(dict,path):
    j = json.dumps(dict)
    f = open(path, "w")
    f.write(j)
    f.close()

def read_tif_meta(fpath):
    with tifffile.TiffFile(fpath) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
    return tif_tags['ImageDescription']

def output_nonz0_crop_backsub(output,hmap,otsu_thresh,path):
    """zeros pixels most in focus at z0,
    crops the worm channel based on the template match path
    subracts the background
    """
    hmap_nonz0 = hmap>1
    open_mask = morphology.opening(hmap_nonz0, morphology.square(3))
    out_open_nonz0 = output*open_mask
    crop_open_output = crop_im(out_open_nonz0,path)
    backsub_crop_open_output = np.fmax(crop_open_output.astype("float")-otsu_thresh,0)
    return backsub_crop_open_output

def output_nonz0_crop_backzero(output,hmap,otsu_thresh,path):
    """zeros pixels most in focus at z0,
    crops the worm channel based on the template match path
    zeros the background pixels with val below the thresh
    """
    hmap_nonz0 = hmap>1
    open_mask = morphology.opening(hmap_nonz0, morphology.square(3))
    out_open_nonz0 = output*open_mask
    crop_open_output = crop_im(out_open_nonz0,path)
    background_mask = crop_open_output>otsu_thresh
    backzero_crop_open_output = background_mask*crop_open_output
    #backsub_crop_open_output = np.fmax(crop_open_output.astype("float")-otsu_thresh,0)
    return backzero_crop_open_output

def conv_8bit_ceiling_bounds(im,ceiling,xrange,yrange):
    im = np.fmin(im,ceiling)/ceiling*255
    return im[xrange[0]:xrange[1],yrange[0]:yrange[1]]

def conv_8bit_ceiling(im,ceiling):
    im = np.fmin(im,ceiling)/ceiling*255
    return im

def get_outline_xy(path):
    chamber_outline = np.vstack((path.vertices,path.vertices[0]))
    xs, ys = zip(*chamber_outline)
    return xs,ys