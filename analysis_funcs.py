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


import struct
import numpy
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

def open_device_dataset_simp(path,file_name):
    #file_name = fname_base + "_A{}_{}h_{}1".format(device, time, sufix)
    data_path = path + "\\" + file_name
    dataset = Dataset(data_path)
    device_meta = {'fname': data_path,
                   'datetime': dataset.summary_metadata['DateAndTime'],
                   'numz': len(dataset.axes['z']),
                   'nump': len(dataset.axes['p']),
                   'numc': len(dataset.axes['channel']),
                   'pos_list': list(dataset.axes['p'])}
    return dataset, device_meta

def open_device_dataset_simp_v2(path,file_name):
    #file_name = fname_base + "_A{}_{}h_{}1".format(device, time, sufix)
    data_path = path + "\\" + file_name
    dataset = Dataset(data_path);
    device_meta = {'fname': data_path,
                   'datetime': dataset.summary_metadata['DateAndTime'],
                   'numz': len(dataset.axes['z']),
                   'nump': len(dataset.axes['p']),
                   'numc': len(dataset.axes['c']),
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

def crop_im(im,path,crop = True,**kwargs):
    # todo: set crop = False so thet image stays fullsize
    return roi.PathROI(path)(im,crop = crop)


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
    strip = np.zeros((2048, length),dtype = np.uint16)
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


def plot_prediction(im, model):
    input_im = np.squeeze(im)
    pred_val = model.predict(input_im, axes='YX')
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(input_im, cmap="magma")
    plt.colorbar(shrink=0.5)
    plt.title('Input');
    plt.subplot(1, 2, 2)
    plt.imshow(pred_val, cmap="magma")
    plt.colorbar(shrink=0.5)
    plt.title('Prediction');


def get_BF_zstack(dataset,pos,numz = 9):
    c = 0
    t = None
    p = pos
    z_stack = np.zeros((numz,2048,2048))
    for z in range(numz):
        z_stack[z,:,:] = dataset.read_image(channel = c,z = z,time = t,p = p)
    return z_stack


def get_BF_im(dataset, pos, z, c=0):
    return dataset.read_image(channel=c, z=z, time=None, p=pos)


def get_RL_zstack(dataset, pos, numz=9, zidx=0):
    c = 1
    t = None
    p = pos
    if zidx == 0:
        z_stack = np.zeros((numz, 2048, 2048), dtype='float32')
        for z in range(numz):
            z_stack[z, :, :] = dataset.read_image(channel=c, z=z, time=t, p=p)
    elif zidx == 2:
        z_stack = np.zeros((2048, 2048, numz), dtype='float32')
        for z in range(numz):
            z_stack[:, :, z] = dataset.read_image(channel=c, z=z, time=t, p=p)
    else:
        raise Exception('Zidx can be 0 or 2')
    return z_stack


def get_zstack(dataset, pos, channel=0, numz=9, zidx=0):
    """zidx = 0 or 2, is for the return shape to be z,rows,cols or rows,cols,z"""
    c = channel
    t = None
    p = pos
    if zidx == 0:
        z_stack = np.zeros((numz, 2048, 2048), dtype='float32')
        for z in range(numz):
            z_stack[z, :, :] = dataset.read_image(channel=c, z=z, time=t, p=p)
    elif zidx == 2:
        z_stack = np.zeros((2048, 2048, numz), dtype='float32')
        for z in range(numz):
            z_stack[:, :, z] = dataset.read_image(channel=c, z=z, time=t, p=p)
    else:
        raise Exception('Zidx can be 0 or 2')
    return z_stack


def to_java_ar(ar, dims):
    """ warning: will not work wityhout jpype"""
    if np.argmin(ar.shape) != 2:
        raise Exception('index of zslices must be 2, for example 2048x2048x9')
    return jpype.JArray(float, dims)(ar)


def plot_before_after(b, a, bounds=None, vminmax=None, colorbar=True, figsize=[12, 6], cmap=None):
    if bounds is None:
        bounds = [0, b.shape[0], 0, b.shape[1]]
    if isinstance(vminmax, list):
        vmin, vmax = (vminmax[0], vminmax[1])
    elif vminmax == 'before':
        vmin, vmax = (np.min(b), np.max(b))
    elif vminmax == 'after':
        vmin, vmax = (np.min(a), np.max(a))
    else:
        vmin, vmax = (None, None)

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(b[bounds[0]:bounds[1], bounds[2]:bounds[3]], vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        plt.colorbar(shrink=0.5)

    plt.subplot(1, 2, 2)
    plt.imshow(a[bounds[0]:bounds[1], bounds[2]:bounds[3]], vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        plt.colorbar(shrink=0.5)


def denoise_zstack(model, zstack, zidx=2):
    output = np.zeros(zstack.shape)
    if zidx == 2:
        for z in range(zstack.shape[2]):
            output[:, :, z] = model.predict(zstack[:, :, z], axes='YX')
    elif zidx == 0:
        for z in range(zstack.shape[0]):
            output[z, :, :] = model.predict(zstack[z, :, :], axes='YX')
    return output


def plot_prediction(im, model):
    input_im = np.squeeze(im)
    pred_val = model.predict(input_im, axes='YX')
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(input_im, cmap="magma")
    plt.title('Input_patch');
    plt.subplot(1, 2, 2)
    plt.imshow(pred_val, cmap="magma")
    plt.title('Prediction_patch');


def run_edf(edf_tools, edf_params, java_stack, shape=(2048, 2048)):
    ''' will not work without jpype'''

    Builder, EdfComplexWavelets, PostProcessing = edf_tools
    im_ware = Builder().create(java_stack)
    edcw = EdfComplexWavelets(edf_params['daubechielength'], edf_params['nScales'], edf_params['subBandCC'],
                              edf_params['majCC'])
    out = edcw.process(im_ware)
    zeros = np.zeros(shape)
    buf = jpype.JArray(float, 2)(zeros)
    out[0].getXY(0, 0, 0, buf)
    out_edf = np.array(buf)

    reasign = PostProcessing.reassignment(out[1], im_ware)
    zeros = np.zeros(shape)
    buf = jpype.JArray(float, 2)(zeros)
    reasign.getXY(0, 0, 0, buf)
    out_hmap = np.array(buf)
    return out_edf, out_hmap


def get_hour_time(time_str_list, Sub=True):
    h_since_E = np.array([time.mktime(time.strptime(h, '%Y/%m/%d %H:%M:%S')) / 60 / 60 for h in time_str_list])
    if not Sub:
        return h_since_E
    else:
        return h_since_E - h_since_E[0]


def get_zstack2(dataset, pos, c=1, channel=0, numz=9, zidx=0):
    """zidx = 0 or 2, is for the return shape to be z,rows,cols or rows,cols,z"""
    # c = channel
    t = None
    p = pos
    if zidx == 0:
        z_stack = np.zeros((numz, 2048, 2048), dtype='float32')
        for z in range(numz):
            z_stack[z, :, :] = dataset.read_image(channel=channel, c=c, z=z, time=t, p=p)
    elif zidx == 2:
        z_stack = np.zeros((2048, 2048, numz), dtype='float32')
        for z in range(numz):
            z_stack[:, :, z] = dataset.read_image(channel=channel, c=c, z=z, time=t, p=p)
    else:
        raise Exception('Zidx can be 0 or 2')
    return z_stack


def get_wstack(dataset, pos, channel=0, cidx=[1, 2, 3], numz=9,dtype = 'float32'):
    """zidx = 0 or 2, is for the return shape to be z,rows,cols or rows,cols,z"""
    # c = channel
    t = None
    p = pos
    # if zidx ==0:
    z_stack = np.zeros((len(cidx), numz, 2048, 2048), dtype=dtype)
    for ci, c in enumerate(cidx):
        for z in range(numz):
            z_stack[ci, z, :, :] = dataset.read_image(channel=channel, c=c, z=z, time=t, p=p)
    #     elif zidx ==2:
    #         z_stack = np.zeros((2048,2048,numz),dtype = 'float32')
    #         for z in range(numz):
    #             z_stack[:,:,z] = dataset.read_image(channel = channel,c=c,z = z,time = t,p = p)
    #     else: raise Exception('Zidx can be 0 or 2')
    return z_stack


def get_crop_mask(path,shape = (2048,2048)):
    return crop_im(np.ones(shape,np.uint8),path)

def create_diagonal_zwall(wstack, mask, chan=0, delta=300):
    # worm_n = stack.shape[0]

    rows = wstack.shape[2]
    cols = wstack.shape[3]
    numz = wstack.shape[1]
    length = 2048 + delta * (numz - 1)
    strip = np.zeros((2048, length))
    for i in range(numz):
        strip[:rows, i * delta:i * delta + cols] += wstack[chan, i, ...] * mask

    return strip


def get_1w_all_times_max_proj(worm_meta, numc=3,start_z = [0,0,0]):
    allt_wstack_proj = np.zeros((len(worm_meta['file_name_list']), numc, 2048, 2048),dtype=np.uint16)

    for timep, file_name in enumerate(worm_meta['file_name_list']):
        dataset, device_meta = open_device_dataset_simp_v2(worm_meta['raw_path'], file_name);
        worm_number = worm_meta['pos_idx']
        wstack = get_wstack(dataset, worm_number, channel=0, cidx=[1, 2, 3], numz=9)

        allt_wstack_proj[timep, 0, ...] = np.max(wstack[0, start_z[0]:, ...], axis=0)
        allt_wstack_proj[timep, 1, ...] = np.max(wstack[1, start_z[1]:, ...], axis=0)
        allt_wstack_proj[timep, 2, ...] = np.max(wstack[2, start_z[2]:, ...], axis=0)

    return allt_wstack_proj

def get_1w_all_times_max_proj_BF_only(worm_meta, numc=1,start_z = [0]):
    allt_wstack_proj = np.zeros((len(worm_meta['file_name_list']), numc, 2048, 2048),dtype=np.uint16)

    for timep, file_name in enumerate(worm_meta['file_name_list']):
        dataset, device_meta = open_device_dataset_simp_v2(worm_meta['raw_path'], file_name);
        worm_number = worm_meta['pos_idx']
        wstack = get_wstack(dataset, worm_number, channel=0, cidx=[1], numz=9)

        allt_wstack_proj[timep, 0, ...] = np.max(wstack[0, start_z[0]:, ...], axis=0)
    return allt_wstack_proj


def get_1w_all_times_denoised_max_proj(worm_meta, numc=3, model_1=None, model_2=None):
    """ WITH DENOISING"""

    allt_wstack_proj = np.zeros((len(worm_meta['file_name_list']), numc, 2048, 2048), dtype=np.uint16)

    for timep, file_name in enumerate(worm_meta['file_name_list']):
        dataset, device_meta = open_device_dataset_simp_v2(worm_meta['raw_path'], file_name);
        worm_number = worm_meta['pos_idx']
        wstack = get_wstack(dataset, worm_number, channel=0, cidx=[1, 2, 3], numz=9)

        allt_wstack_proj[timep, 0, ...] = wstack[0, 2, ...]
        allt_wstack_proj[timep, 1, ...] = np.max(denoise_zstack(model_1, wstack[1, :, ...], zidx=0)
                                                 , axis=0)
        allt_wstack_proj[timep, 2, ...] = np.max(denoise_zstack(model_2, wstack[2, :, ...], zidx=0)
                                                 , axis=0)

    return allt_wstack_proj


def get_cropped_all_times_max_proj(all_t_stack, center, verts, worm_meta=None):
    cropped_all_t_stack = np.zeros(all_t_stack.shape, dtype=np.uint16)
    for timep in range(all_t_stack.shape[0]):
        path, _ = find_roi(all_t_stack[timep, 0, ...], center, verts)
        xmin, xmax = (int(np.min(path.vertices[:, 0])), int(np.max(path.vertices[:, 0])))
        ymin, ymax = (int(np.min(path.vertices[:, 1])), int(np.max(path.vertices[:, 1])))
        for cc in range(all_t_stack.shape[1]):
            cropped_all_t_stack[timep, cc, ymin:ymax, xmin:xmax] = crop_im(all_t_stack[timep, cc, ...], path)
    return cropped_all_t_stack

def get_cropped_all_times_max_proj_v2(all_t_stack, center, verts, worm_meta=None):
    cropped_all_t_stack = np.zeros(all_t_stack.shape, dtype=np.uint16)
    for timep in range(all_t_stack.shape[0]):
        path, _ = find_roi(all_t_stack[timep, 0, ...], center, verts)
        # xmin, xmax = (int(np.min(path.vertices[:, 0])), int(np.max(path.vertices[:, 0])))
        # ymin, ymax = (int(np.min(path.vertices[:, 1])), int(np.max(path.vertices[:, 1])))
        for cc in range(all_t_stack.shape[1]):
            cropped_all_t_stack[timep, cc, ...] = crop_im(all_t_stack[timep, cc, ...], path,crop = False)
    return cropped_all_t_stack

def get_cropped_all_times_max_proj_v3(all_t_stack, center, verts, worm_meta=None):
    cropped_all_t_stack = np.zeros(all_t_stack.shape, dtype=np.uint16)
    path = None
    for timep in range(all_t_stack.shape[0]):
        if path is None:
            path, _ = find_roi(all_t_stack[timep, 0, ...], center, verts)
        for cc in range(all_t_stack.shape[1]):
            cropped_all_t_stack[timep, cc, ...] = crop_im(all_t_stack[timep, cc, ...], path,crop = False)
    return cropped_all_t_stack


def get_cropped_all_times_max_proj_v4(all_t_stack, center, verts, worm_meta=None):
    path, _ = find_roi(all_t_stack[0, 0, ...], center, verts)
    c_mask_c = np.ones((2048, 2048), dtype=np.uint16)
    c_mask_c = crop_im(c_mask_c, path, crop=False)
    all_t_stack = all_t_stack * c_mask_c[np.newaxis, np.newaxis, ...]
    return all_t_stack


def get_cropped_all_times_max_proj_back_sub_first_median(all_t_stack, center, verts, worm_meta=None):
    """ WITH BACKGROUND SUBTRACTION USING THE MEDIAN PIX INTENSITY OF WORM CROP PRE EXPERIMENT"""
    cropped_all_t_stack = np.zeros(all_t_stack.shape, dtype=np.uint16)
    med_background = None
    for timep in range(all_t_stack.shape[0]):
        path, _ = find_roi(all_t_stack[timep, 0, ...], center, verts)
        xmin, xmax = (int(np.min(path.vertices[:, 0])), int(np.max(path.vertices[:, 0])))
        ymin, ymax = (int(np.min(path.vertices[:, 1])), int(np.max(path.vertices[:, 1])))
        for cc in range(all_t_stack.shape[1]):
            crop = crop_im(all_t_stack[timep, cc, ...], path)
            if cc > 0:
                if med_background is None:
                    med_background = np.median(crop[np.nonzero(crop)])
                crop = np.fmax(0, crop - med_background)
            cropped_all_t_stack[timep, cc, ymin:ymax, xmin:xmax] = crop
    return cropped_all_t_stack


def create_diagonal_posstrip_multi_chan(stack, delta=300):
    worm_n = stack.shape[0]
    numc = stack.shape[1]
    length = 2048 + delta * (worm_n - 1)
    strip = np.zeros((numc, 2048, length), dtype=np.uint16)
    # sum_projection = np.zeros((crop_im.shape[0], crop_im.shape[1]))
    for cc in range(numc):
        for i in range(worm_n):
            image = np.squeeze(stack[i, cc, :, :])
            rows = image.shape[0]
            cols = image.shape[1]
            strip[cc, :rows, i * delta:i * delta + cols] += image
    return strip


def imagej_metadata_tags(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    """
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder + ('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder + 'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder + ('I' * len(bytecounts)), *bytecounts)
    return ((50839, 'B', len(data), data, True),
            (50838, 'I', len(bytecounts) // 4, bytecounts, True))


def get_grayredgreen_lut():
    grays = numpy.tile(numpy.arange(256, dtype='uint8'), (3, 1))
    red = numpy.zeros((3, 256), dtype='uint8')
    red[0] = numpy.arange(256, dtype='uint8')
    green = numpy.zeros((3, 256), dtype='uint8')
    green[1] = numpy.arange(256, dtype='uint8')
    return [grays, red, green]

def get_grayredgreenblue_lut():
    grays = numpy.tile(numpy.arange(256, dtype='uint8'), (3, 1))
    red = numpy.zeros((3, 256), dtype='uint8')
    red[0] = numpy.arange(256, dtype='uint8')
    green = numpy.zeros((3, 256), dtype='uint8')
    green[1] = numpy.arange(256, dtype='uint8')
    blue = numpy.zeros((3, 256), dtype='uint8')
    blue[2] = numpy.arange(256, dtype='uint8')
    return [grays, red, green,blue]

def get_grayredgreenbluemagenta_lut():
    grays = numpy.tile(numpy.arange(256, dtype='uint8'), (3, 1))
    red = numpy.zeros((3, 256), dtype='uint8')
    red[0] = numpy.arange(256, dtype='uint8')
    green = numpy.zeros((3, 256), dtype='uint8')
    green[1] = numpy.arange(256, dtype='uint8')
    blue = numpy.zeros((3, 256), dtype='uint8')
    blue[2] = numpy.arange(256, dtype='uint8')
    magenta = numpy.zeros((3, 256), dtype='uint8')
    magenta[0] = numpy.arange(256, dtype='uint8')
    magenta[2] = numpy.arange(256, dtype='uint8')
    return [grays, red, green,blue,magenta]

# def get_ranges(array):
#     if len(array.shape)==4:
#         mx = np.max(array,axis = (-4,-2,-1)).astype('float')
#         mn = np.min(array,axis = (-4,-2,-1)).astype('float')
#     elif len(array.shape)==3:
#         mx = np.max(array,axis = (-2,-1)).astype('float')
#         mn = np.min(array,axis = (-2,-1)).astype('float')
#     else:
#         raise IndexError('array must have 3 or 4 inedex dims to use get ranges')
#     ranges = [[n,x] for n,x in zip(mn,mx)]
#     print(type(ranges[0][0]))
#     return ranges


def save_tiff_ijmeta_grayredgreen(fname, array):
    # ranges does not work, some problem with write doubles so i cnat base in a list of lists with min max for each channel
    # try:
    #     ijtags = imagej_metadata_tags({'LUTs': get_grayredgreen_lut(),
    #                                    'Ranges':get_ranges(array)}, '>')
    # except IndexError:
    #    print('could not call get_rnages. saving without ranges'
    ijtags = imagej_metadata_tags({'LUTs': get_grayredgreen_lut()}, '>')
    tifffile.imsave(fname, array, byteorder='>', imagej=True,
                    metadata={'mode': 'color'}, extratags=ijtags)

def save_tiff_ijmeta_grayredgreenblue(fname, array):
    ijtags = imagej_metadata_tags({'LUTs': get_grayredgreenblue_lut()}, '>')
    tifffile.imsave(fname, array, byteorder='>', imagej=True,
                    metadata={'mode': 'color'}, extratags=ijtags)


def save_tiff_ijmeta_grayredgreenbluemagenta(fname, array):
    ijtags = imagej_metadata_tags({'LUTs': get_grayredgreenbluemagenta_lut()}, '>')
    tifffile.imsave(fname, array, byteorder='>', imagej=True,
                    metadata={'mode': 'color'}, extratags=ijtags)