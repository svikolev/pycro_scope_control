import numpy as np
import time
import json
import cv2
import matplotlib.pyplot as plt
import pickle

import sys
import copy
import os
print(sys.path)
import datetime

from pycromanager import Acquisition, multi_d_acquisition_events, Dataset, Core, Studio# , Bridge

import pycro_scope_control
#from pycro_scope_control import *
import pycro_scope_control.analysis_funcs as af

from scipy.fft import fft2, fftshift
#from scipy.ndimage import imread
import os


from skimage import morphology, exposure, filters, measure
from scipy import signal, ndimage
import copy
from scipy import stats

#import analysis_funcs as afS

import heapq
import threading


def get_all(java_str_vec):
    """returns a python list from a java str vector"""
    i = 0
    ar = []
    while True:
        try:
            ar.append(java_str_vec.get(i))
            i += 1
        except:
            break
    return ar


def get_properties_dict(core, dev_name, print_dict=False):
    """returns the properties dict of a device, prop:value"""
    prop_names = get_all(core.get_device_property_names(dev_name))
    prop_dict = {p_name: core.get_property(dev_name, p_name) for p_name in prop_names}
    if print_dict:
        for key, value in prop_dict.items():
            print(key, '   :   ', value)
    return prop_dict


def java_vector(bridge,array):
    array_java = bridge._construct_java_object("mmcorej.DoubleVector")
    for i in array:
        array_java.add(float(i))
    return array_java

def java_str_vector(bridge,str_array):
    array_java = bridge._construct_java_object("mmcorej.StrVector")
    for i in str_array:
        array_java.add(str(i))
    return array_java


def upload_triangle_z_seq(core, bridge, pos_sequence):
    """sends sequence to piezo stage
    needs core, bridge, and pos_sequence list
    """
    z_stage = 'ZStage'
    # core.get_focus_device() # just a string of device name, will be 'ZStage'
    # create java obj with the seq
    pos_seq_java = java_vector(bridge,pos_sequence)

    core.set_property(z_stage, "Use Sequence", "Yes")
    core.set_property(z_stage, "Use Fast Sequence", "No")
    core.load_stage_sequence(z_stage, pos_seq_java)
    core.set_property(z_stage, "Use Fast Sequence", "Armed")

    return None


def upload_triangle_z_seq_plus(core, bridge, pos_sequence, set_zero = False, dup_last = False,dupL_num = 1, dup_first = False, dupF_num = 1):
    """ ***plus version can send a duplicate at the first or last pos
    sends sequence to piezo stage
    needs core, bridge, and pos_sequence list
    optional: add extra pos at the end
        since sequencing skips pos when using micromanager event stream
        alternative is to disarm, then arm the 'ZStage' sequencer"""
    z_stage = 'ZStage'
    if set_zero:
        core.set_position(z_stage)
    # core.get_focus_device() # just a string of device name, will be 'ZStage'
    # create java obj with the seq
    pos_seq_java = bridge._construct_java_object("mmcorej.DoubleVector")
    if dup_first:
        for i in range(dupF_num):
            pos_seq_java.add(float(pos_sequence[0]))
    for i in pos_sequence:
        pos_seq_java.add(float(i))
    if dup_last:
        for i in range(dupL_num):
            pos_seq_java.add(float(pos_sequence[-1]))

    core.set_property(z_stage, "Use Sequence", "Yes")
    core.set_property(z_stage, "Use Fast Sequence", "No")
    core.load_stage_sequence(z_stage, pos_seq_java)
    core.set_property(z_stage, "Use Fast Sequence", "Armed")

    return get_all(pos_seq_java)


def create_triangle_z_seq_doubles(start_pos, mid_pos, step_size):
    single_seq = np.hstack(
        (np.arange(start_pos, mid_pos + step_size, step_size),
         np.arange(mid_pos, start_pos - step_size, -step_size)[1:],))
    double_seq = np.vstack((single_seq, single_seq))
    pos_sequence = double_seq.reshape(len(single_seq) * 2, order='F')

    return pos_sequence


def create_triangle_z_seq_singles(start_pos, mid_pos, step_size):
    single_seq = np.hstack(
        (np.arange(start_pos, mid_pos + step_size, step_size),
         np.arange(mid_pos, start_pos - step_size, -step_size),))
    return single_seq


def create_triangle_z_seq_triples(start_pos, mid_pos, step_size):
    single_seq = np.hstack(
        (np.arange(start_pos, mid_pos + step_size, step_size),
         np.arange(mid_pos, start_pos - step_size, -step_size)[1:],))
    double_seq = np.vstack((single_seq, single_seq, single_seq))
    pos_sequence = double_seq.reshape(len(single_seq) * double_seq.shape[0], order='F')
    return pos_sequence


def create_straight_z_seq_doubles(start_pos, mid_pos, step_size):
    single_seq = np.arange(start_pos, mid_pos + step_size, step_size)
    double_seq = np.vstack((single_seq, single_seq))
    pos_sequence = double_seq.reshape(len(single_seq) * 2, order='F')
    return pos_sequence


def create_straight_z_seq_singles(start_pos, mid_pos, step_size):
    single_seq = np.arange(start_pos, mid_pos + step_size, step_size)
    return single_seq  # pos_sequence


def go_to_xyz(core, xyz):
    core.set_xy_position(xyz[0], xyz[1])
    core.set_position('ZeissFocusAxis', xyz[2])
    core.wait_for_device('ZeissFocusAxis')
    core.wait_for_device('XYStage')


def shutter_LED(c, o=False):
    c.set_shutter_open('ZeissColibri', o)

def shutter_TL(c, o=False):
    c.set_shutter_open('ZeissTransmittedLightShutter', o)

def shutter_RL(c,o=False):
    c.set_shutter_open('ZeissReflectedLightShutter', o)


def go_to_xyz_careful(core, xyz,sleep = 0):
    """ moves to a xyz position carefully. the carefully refers to the z pos.
    code checks to see if the new z will be greater thank or less than the current and then moves
    either the z or the xy first, sleep after moving the xy because this takes a long time sometimes and the stage may timeout

    *** should in future increase the timeout time of the stage in the MM device properties ***
    """
    cur_z = core.get_position('ZeissFocusAxis')
    if xyz[2] > cur_z:
        core.set_xy_position(xyz[0], xyz[1])
        if sleep > 0:
            time.sleep(sleep)
        core.wait_for_device('XYStage')
        core.set_position('ZeissFocusAxis', xyz[2])
        core.wait_for_device('ZeissFocusAxis')
    else:
        core.set_position('ZeissFocusAxis', xyz[2])
        core.wait_for_device('ZeissFocusAxis')
        core.set_xy_position(xyz[0], xyz[1])
        if sleep > 0:
            time.sleep(sleep)
        core.wait_for_device('XYStage')


def compute_rotation_trans(dxy,loc):
    """computes rotation transformation"""
    dx = dxy[0]
    dy = dxy[1]

    xloc = loc[0] -500
    yloc = loc[1] -500
    a = np.array([[xloc, -yloc], [yloc, xloc]])
    b = np.array([dx, dy])
    x = np.linalg.solve(a, b)
    return  x #np.array([[x[, -yloc], [yloc, xloc]])


def get_current_pos_dict(c,mode = 'all'):
    """gets dictionary of the current positiion,
    mode = all:
        {X,Y,ZStage,ZeissFocusAxis}
    mode = xyz:
        {X,Y,ZStage,Z}"""
    pos = {}
    if mode =='all':
        pos['x'] = c.get_x_position("XYStage")
        pos['y'] = c.get_y_position("XYStage")
        pos['ZStage'] = c.get_position("ZStage")
        pos['ZeissFocusAxis'] = c.get_position('ZeissFocusAxis')
        return pos
    elif mode == 'xyz':
        pos['X'] = c.get_x_position("XYStage")
        pos['Y'] = c.get_y_position("XYStage")
        pos['ZStage'] = c.get_position("ZStage")
        pos['Z'] = c.get_position('ZeissFocusAxis')
        return pos
    else:
        raise ValueError("mode can be all, or xyz only")


def get_current_pos_list(c):
    """returns a list of [x,y,zpiezo,z-nose_peice]"""
    pos = [c.get_x_position("XYStage"),c.get_y_position("XYStage"),c.get_position("ZStage"),c.get_position('ZeissFocusAxis')]
    return pos

def get_pos_list_from_manager(studio):
    """gets the position list from Micromager position manger gui
    only gets the x,y,z positions
    for z pos it uses the zeiss focus axis"""
    position_list_manager = studio.get_position_list_manager()
    pos_list = position_list_manager.get_position_list()

    xyz_ar = []
    for i in range(pos_list.get_number_of_positions()):
        posi = pos_list.get_position(i)
        for j in range(posi.size()):
            stage_d = posi.get(j)
            if stage_d.stageName == 'ZeissFocusAxis':
                z = stage_d.x
            if stage_d.stageName == 'XYStage':
                x = stage_d.x
                y = stage_d.y
        xyz_ar.append([x, y, z])

    return xyz_ar


def remove_excluded_pos(pos_list, exclude_list):
    """takes a position list of dictionaries, and returns the same list without the dictionaries where used is false"""
    out_pos_list = []
    for pos_dict in pos_list:
        if int(pos_dict['Name'][1:]) in exclude_list:
            pos_dict['Used'] = False
        out_pos_list.append(pos_dict)
    return out_pos_list


def compute_device_tiles(edge_ar, num_tiles, L_to_R=True, z_mode='inter_center', offset=15, dev_name='A1'):
    ##### adapted from other file, docstring is not yet verified##

    """ Takes in a single device dictionary with the xyz of the left and right tiles
    and the number of tiles to be positioned, and mode for z position and returns a list of tile positions and the center of the array
    PARAMETERS
    ----------
    1) device_dict --> dictionary of the edge positions
    2) num_tiles   --> number of tiles to insert
    3) L_to_R      --> bool if the edges are [left,right], false if [right,left]
    3) Z_modes     --> descrete options for selecting z positions
        *default='inter_center' *all use the x axis to spread the z pos
        a) inter_center        --> interpolate a single center z position from the edges
        b) inter_all           --> interpolate all z positions from edges
        c) inter_center_offset --> interpolate a single Z pos and add offset (for zstack if input is bottom focus
        d) inter_all_offset    --> interpolate all z positions from edges and add offset (for zstack if input is bottom)
    4) offset     --> offset distance for z_modes c)/d) when using a zstack and the unput tile z_pos is the surface of the slide
        * Default is +15um
    5) dev name   --> name of the device such as A1 or A2 or C4...

    computes distance between edges and devides it into n-1 sections.
    xy for each tile are distributed evenly in this range
    center of array is computed by left side + half the distance between edges

    RETURNS
    -----------
    1) a list of dictionaries of tiles xyz pos distributed between the edges
    2) center_x of the array
    3) center_y of the array"""

    tiles = []
    # num_tiles = 8
    # start_x = float(device_dict['left']['X'])
    start_x = edge_ar[0][0]
    device_dist_x = (edge_ar[1][0]) - start_x
    delta_x = device_dist_x / (num_tiles - 1)
    center_x = round(start_x + device_dist_x / 2, 3)

    # start_z = float(device_dict['left']['Z'])
    start_z = edge_ar[0][2]
    device_dist_z = (edge_ar[1][2]) - start_z
    # device_dist_z = (float(device_dict['right']['Z'])-start_z)
    delta_z = device_dist_z / (num_tiles - 1)
    center_z = round(start_z + device_dist_z / 2, 3)

    # start_y = float(device_dict['left']['Y'])
    start_y = edge_ar[0][1]
    device_dist_y = (edge_ar[1][1]) - start_y
    # device_dist_y = abs((float(device_dict['right']['Y'])-start_y))
    delta_y = device_dist_y / (num_tiles - 1)
    center_y = round(start_y + device_dist_y / 2, 3)

    for i in range(num_tiles):
        x_pos = round(start_x + delta_x * i, 3)
        y_pos = round(start_y + delta_y * i, 3)
        if z_mode == 'inter_center':
            z_pos = center_z
        elif z_mode == 'inter_center_offset':
            z_pos = center_z + offset
        elif z_mode == 'inter_all':
            z_pos = round(start_z + delta_z * i, 3)
        elif z_mode == 'inter_all_offset':
            z_pos = round(start_z + delta_z * i, 3) + offset

        if L_to_R:
            tiles.append(
                {'Name': 'P{}'.format(1 + i), 'X': x_pos, 'Y': y_pos, 'Z': z_pos, 'Used': True, 'Dev': dev_name})
        else:
            tiles.append({'Name': 'P{}'.format(num_tiles - i), 'X': x_pos, 'Y': y_pos, 'Z': z_pos, 'Used': True,
                          'Dev': dev_name})

    return tiles, center_x, center_y

def set_pre_acq_params(core,exposure_t = 30,cube = 'semrock_TBP',chan_group = 'Channel'):
    """ acquisition params"""

    # set up camera config ###
    core.set_config('Flash4_Triggers', 'Out_pos_exp')
    core.wait_for_config('Flash4_Triggers', 'Out_pos_exp')
    core.set_exposure(exposure_t)
    # set up camera config ###

    # set up TL config ###
    # core.set_config('TL_lamp', 'BF_2_5')
    # core.wait_for_config('TL_lamp', 'BF_2_5')
    core.set_config('TL_lamp', 'BF_20x_25v_55na_5bf')
    core.wait_for_config('TL_lamp', 'BF_20x_25v_55na_5bf')
    # set up TL config ###

    # set up cube config ###
    core.set_config('Cubes', cube)
    core.wait_for_config('Cubes', cube)
    # set up cube config ###

    # set up Light Path config ###
    core.set_config('Light_Path', 'R_Cam')
    core.wait_for_config('Light_Path', 'R_Cam')
    # set up Light Path config ###

    core.set_channel_group(chan_group)
    core.set_position('ZStage', 0)
    ### check colibri shutter is main shutter and remains open, and autoshutter is off##

    core.set_shutter_device('Arduino-Shutter')
    core.set_auto_shutter(False)

def check_acq_configs():
    """need to add this later"""
    print("testing reImport")

def events_TL_multi_pos(pos_list,offset = 0,channelgoup = 'LEDs_wheel_chan',chan = 'TL_noF'):
    """ takes list of pos dictionaies and creates a list of TL events
    the axis is just """
    #pos_list = B1
    events = []
    im_num = 0
    for pos_num, pos_dict in enumerate(pos_list):
        if pos_dict['Used']:
            evt = { 'axes': {'im':im_num},
                   'channel': {'group': 'LEDs_wheel_chan', 'config': 'TL_noF'},
                   'x': pos_dict['X'],'y': pos_dict['Y'],'z': pos_dict['Z']+offset}
            events.append(evt)
            im_num +=1
    return events


def include_all_pos(pos_list):
    """returns a new list of dicts of positions with all 'Used' set true """
    out_pos_list = []
    for pos_dict in pos_list:
        dict2 = pos_dict.copy()
        dict2['Used'] = True
        out_pos_list.append(dict2)
    return out_pos_list


def pos_list_whole_strip(arr_edges, exclude_LofL=None, dev_tiles=32, dev_names_list=['A1'], L_to_R=True,
                         z_mode='inter_all'):
    """takes in a list of xyz pos for the first and last worm in each device
    returns a list of lists (a list for each device) with the positions of each worm
    arr_egde must have even length"""
    strip_pos_LofL = []
    for j, i in enumerate(range(0, len(arr_edges), 2)):
        Dev_i, _, _ = compute_device_tiles(arr_edges[i:i + 2], dev_tiles, L_to_R, z_mode, dev_name=dev_names_list[j])
        if exclude_LofL is not None:
            Dev_i = remove_excluded_pos(Dev_i, exclude_LofL[j])
        strip_pos_LofL.append(Dev_i)
    return strip_pos_LofL


def create_events_multi_pos_multi_chan_zstack(pos_list, z_idx, num_time_points, RL_chan = '505', include_all = False, z_offset = 0):
    """legacy method as of march 23, does not use fast sequncing"""
    events = []
    z_idx_ = z_idx.copy()
    for pos_num, pos_dict in enumerate(pos_list):
        if pos_dict['Used'] or include_all:
            events.append({"axes": {"p": pos_num, "time": 0, "z": 0},
                           'channel': {'group': 'Channel', 'config': 'TL'},
                           'x': pos_dict['X'], 'y': pos_dict['Y'], 'z': pos_dict['Z'] + z_offset})

            for i in range(num_time_points):
                for j in z_idx_:
                    events.append({"axes": {"p": pos_num, "time": i + 1, "z": j},
                                   'channel': {'group': 'Channel', 'config': RL_chan}})
                z_idx_.reverse()
    return events


def create_events_multi_pos_multi_chan_same_zstack(pos_list, num_z, RL_chan = '505', include_all = False, z_offset = 0):
    """legacy method as of march 23,does not use fast sequncing"""
    events = []
    for pos_num, pos_dict in enumerate(pos_list):
        if pos_dict['Used'] or include_all:
            for i in range(num_z):
                events.append({"axes": {"p": pos_num, "z": i},
                               'channel': {'group': 'Channel', 'config': 'TL'},
                               'x': pos_dict['X'], 'y': pos_dict['Y'], 'z': pos_dict['Z'] + z_offset})
            for i in range(num_z):
                events.append({"axes": {"p": pos_num, "z": i},
                               'channel': {'group': 'Channel', 'config': RL_chan}})
    return events

def reset_piezo(core):
    """stops piezo sequencing, sets pos to 0 and returns pos just be sure"""
    core.set_property('ZStage', "Use Fast Sequence", "No")
    core.set_property('ZStage', "Use Sequence", "No")
    core.set_position('ZStage', 0)
    return core.get_position('ZStage')


def prep_for_seq_imaging(core,bridge,pos_sequence,autoshutter = True,piezo_start = -5):
    """legacy method as of march 2023"""
    core.set_focus_device('ZeissFocusAxis')
    core.set_auto_shutter(autoshutter)
    core.set_position('ZStage', piezo_start)
    upload_triangle_z_seq(core, bridge, pos_sequence)
    return None


def get_lefmost_pos(pLofL):
    return [pLofL[0][0]["X"],pLofL[0][0]["Y"],pLofL[0][0]["Z"]]

#
# def wait_for_file(file_str, sleeptime = 60, timeout_time = 60*60*2):
#     waitedtime = 0
#     time_yet = False
#     while not time_yet:
#         time_yet = os.path.exists(file_str)
#         if not time_yet:
#             time.sleep(sleeptime)
#             waitedtime += sleeptime
#         assert(waitedtime < timeout_time,"waited full timeout time {} sec".format(timeout_time))

def find_chamber(wormchamber, template):
    """opn cv template match. takes in the image and the template,
    returns the x,y pos of the top left of the image
    this implementation """
    res2 = cv2.matchTemplate(wormchamber.astype(np.float32), template.astype(np.float32), method=cv2.TM_CCOEFF);
    (_, _, _, maxLoc) = cv2.minMaxLoc(res2)
    return maxLoc


def adjust_edge_ar(dataset,center,edge_ar,trans_mat):
    """ takes in a dataset of multi position BF images and the array of the xyz positions they were taken at.
    uses the images and template to find the center of the channel and then adjestz the positions to be more accurately in the center
    returns the adjusted positions"""
    image_stack = np.array(np.squeeze(dataset.as_array()))
    adjusted_edge_ar = copy.deepcopy(edge_ar)
    #image_stack.shape
    for i in range(image_stack.shape[0]):
        loc = np.array(find_chamber(image_stack[i,:,:],center))-500
        v_move = -np.matmul(trans_mat,loc)
        adjusted_edge_ar[i][0] +=v_move[0]
        adjusted_edge_ar[i][1] +=v_move[1]
    return adjusted_edge_ar

def plot_position_list(edge_ar,marker = '*'):
    z = [p[2] for p in edge_ar]
    x = [p[0] for p in edge_ar]
    y = [p[1] for p in edge_ar]

    plt.subplot(2,1,1)
    plt.plot(x,z,marker = marker)
    plt.title('X v Z')
    plt.subplot(2,1,2)
    plt.title('X v Y')
    plt.plot(x,y,marker = marker)

def save_json(dict,path):
    j = json.dumps(dict)
    f = open(path, "w")
    f.write(j)
    f.close()

def write_pos_file(pos_list,path):
    """does not work"""
    pos_file = {
      "encoding": "UTF-8",
      "format": "Micro-Manager Property Map",
      "major_version": 2,
      "minor_version": 0,
      "map": {
        "StagePositions": {
          "type": "PROPERTY_MAP",
          "array": []
        }
      }
    }
    for pos_d in pos_list:
        pos_file["map"]["array"].append([
            {
              "DefaultXYStage": {
                "type": "STRING",
                "scalar": "XYStage"
              },
              "DefaultZStage": {
                "type": "STRING",
                "scalar": "ZStage"
              },
              "DevicePositions": {
                "type": "PROPERTY_MAP",
                "array": [
                  {
                    "Device": {
                      "type": "STRING",
                      "scalar": "ZStage"
                    },
                    "Position_um": {
                      "type": "DOUBLE",
                      "array": [
                        pos_d['ZStage']
                      ]
                    }
                  },
                  {
                    "Device": {
                      "type": "STRING",
                      "scalar": "ZeissFocusAxis"
                    },
                    "Position_um": {
                      "type": "DOUBLE",
                      "array": [
                        pos_d['ZeissFocusAxis']
                      ]
                    }
                  },
                  {
                    "Device": {
                      "type": "STRING",
                      "scalar": "ZeissDefiniteFocusOffset"
                    },
                    "Position_um": {
                      "type": "DOUBLE",
                      "array": [
                        0.0
                      ]
                    }
                  },
                  {
                    "Device": {
                      "type": "STRING",
                      "scalar": "XYStage"
                    },
                    "Position_um": {
                      "type": "DOUBLE",
                      "array": [
                        pos_d['x'],
                        pos_d['y']
                      ]
                    }
                  }
                ]
              },
              "GridCol": {
                "type": "INTEGER",
                "scalar": 0
              },
              "GridRow": {
                "type": "INTEGER",
                "scalar": 0
              },
              "Label": {
                "type": "STRING",
                "scalar": "Pos0"
              },
              "Properties": {
                "type": "PROPERTY_MAP",
                "scalar": {}
              }
            }
          ])
    save_json(pos_file,path)

def image_single_focusing_zstack(c,pos,zrange,zstep,chan="TL_noF",file_name = 'focusing_zstack',
                                 data_path = r"C:\Users\LevineLab\Documents\python notebooks",display = True):
    """ code to image single zstacks in a single channel, used the zeiss focus axis as the focus device
        could change this to use the piezo so that its faster:
        piezo has -50-50um range and is usually at 0 so need to incorporate shifting the startz up by 50 if i want to take zstack greater then 50um

        note: sets autoshutter"""
    events = []
    im = 0
    #for im, pos in enumerate(out_chan_ar):
    for j, z in enumerate(range(zrange[0], zrange[1], zstep)):
        events.append({'axes': {'im': im, 'z': j}, 'channel': {'group': 'LEDs_wheel_chan', 'config': chan},
                       'x': pos[0], 'y': pos[1], 'z': pos[2] + z})

    c.set_focus_device('ZeissFocusAxis')  # 'ZStage')
    c.set_config('LEDs_wheel_chan', chan)
    c.set_auto_shutter(True)

    #file_name = "TL_alignment_zstack_right_to_left"
    with Acquisition(directory=data_path, name=file_name, show_display=display, debug=False) as acq:
        acq.acquire(events)
        zstack_dataset = acq.get_dataset()
    return zstack_dataset


def image_multi_focusing_zstack(c,pos_array,zrange,zstep,chan="TL_noF",file_name = 'focusing_zstack',
                                data_path = r"C:\Users\LevineLab\Documents\python notebooks",display = True):
    """ code to image multiple zstacks in a single channel, used the zeiss focus axis as the focus device
    could change this to use the piezo so that its faster:
    piezo has -50-50um range and is usually at 0 so need to incorporate shifting the startz up by 50 if i want to take zstack greater then 50um

    note: sets autoshutter"""
    events = []
    im = 0
    for im, pos in enumerate(pos_array):
        for j, z in enumerate(range(zrange[0], zrange[1], zstep)):
            events.append({'axes': {'im': im, 'z': j}, 'channel': {'group': 'LEDs_wheel_chan', 'config': chan},
                           'x': pos[0], 'y': pos[1], 'z': pos[2] + z})

    c.set_focus_device('ZeissFocusAxis')  # 'ZStage')
    c.set_config('LEDs_wheel_chan', chan)
    c.set_auto_shutter(True)

    #file_name = "TL_alignment_zstack_right_to_left"
    with Acquisition(directory=data_path, name=file_name, show_display=display, debug=False) as acq:
        acq.acquire(events)
        zstack_dataset = acq.get_dataset()
    return zstack_dataset


def radial_average(data, center=None):
    """ radially averages an image from -pi to pi
    currently returns a vector that is half diagonal long. I should change this so that it cuts the corners"""
    y, x = np.indices((data.shape))
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    r = r.astype(int)
    radial_mean = np.bincount(r.ravel(), data.ravel()) / np.bincount(r.ravel())
    return radial_mean

def radial_mean(image):
    """takes in an image, computes FFT, shifts FFT, gets power spectrum (magnitude), the radially averages to get 1d vector
    I think this vector coresponds to increasing frequency pressence in the image"""
    #image = imread(image_path, mode='F')
    image = (image - image.mean()) / image.std()
    f_image = fftshift(fft2(image))
    log_power_spectrum = np.log(np.abs(f_image) ** 2)
    radial_mean = radial_average(log_power_spectrum)
    #high_freq_sum = radial_mean[int(radial_mean.size / 2):].sum()
    return radial_mean #high_freq_sum


def first_greater(arr, target):
    for idx, value in enumerate(arr):
        if value > target:
            return idx, value
    return None, None


def find_increasing_values(data, threshold=0.1):
    """greedy algorithm to find the top of a cliff of a function that is exponential to a point then is varaible"""
    previous_value = data[0]
    increasing_values = []

    for index, value in enumerate(data[1:], start=1):
        if value >= (1 + threshold) * previous_value:
            increasing_values.append((index, value))
        elif value > 0.5 and increasing_values:
            break
        previous_value = value

    return increasing_values, index - 1

def get_ard_seq_syncexp(ard_channels,ard_wheel,gap_frames,numz):
    """ generates the sequence of arduino trigger patterns
        Arduino firmware edited to respond to both rising edge and falling edge triggers
        adds a 0 for null frame at start,
        then loops over the channels and loops over the zstack to add repeats so same LED is on for entire zstack,
        adds wheel trig after each zstack
        adds the gap frames but only if it is not the last channel
        Parameters:
        -------------
        ard_channels -> list of the binary pattern for the LED channels
        ard_wheel -> ard channel that corresponds to the wheel trigger
        gap_frames -> number of gaps
        numz
        """
    ard_seq = []
    ard_seq.append(0)
    for i,chan in enumerate(ard_channels):
        for _ in range(numz*2+1):
            ard_seq.append(chan)
        ard_seq.append(ard_wheel)
        if i<len(ard_channels)-1:
            for _ in range((gap_frames-1)*2):
                ard_seq.append(0)
        else:
            ard_seq.append(0)
    return ard_seq

def get_zseq_sycexp(numz,start_z,step_z,ard_channels,gap_frames = 3):
    """ generates the sequence of z positions
    adds a 0 for null frame at start, the loops over the channels and adds in the zstack, adds 0s for the null frames
    as the gap frames but only if it is not the last channel
    Parameters:
    -------------
    numz
    startz
    stepz
    ard_channels -> list of the binary pattern for the LED channels
    """
    end_z  = start_z+step_z*numz
    z_pos = np.arange(start_z,start_z+numz*step_z,step_z)
    zseq = []

    zseq.append(z_pos[0])
    for i,_ in enumerate(ard_channels):
        zseq+=list(z_pos[1:])
        zseq.append(z_pos[-1])
        zseq.append(z_pos[0])
        if i<len(ard_channels)-1:
            for g in range(gap_frames-1):
                zseq.append(z_pos[0])
                zseq.append(z_pos[0])
    return zseq

def get_fast_imaging_seq_set_1(ard_channels = [1, 2, 4], ard_wheel = 8,gap_frames = 3,numz = 9,start_z = 0,step_z = 5,pin16 = False):
    """ retrieves the set of ard and z sequances for imaging multichannel multi z
    calls code to:
    1) generates the list of trigger patterns for the arduino
    2) generate list of trigger z positions for the piezo stage

    Parameters:
    -------------
    ard_channels -> list of the binary pattern for the LED channels
    ard_wheel -> ard channel that corresponds to triggering the wheel
    gap_frames -> number of gaps between channels
    numz -> number of z stack pos
    startz -> start pos of the piezo
    step_z -> size of z steps in um
    pin_16 -> optional to add ard channel 16 on for each pattern that has LED on. this is used for measuring the triggers timing of the sequencing"""
    ard_seq = get_ard_seq_syncexp(ard_channels, ard_wheel, gap_frames, numz)
    if pin16:
        ard_seq = [s + 16 if s != 0 else s for s in ard_seq]
    zseq = get_zseq_sycexp(numz, start_z, step_z, ard_channels)
    #state_seq_java = pf.java_str_vector(b, ard_seq[:])
    return ard_seq,zseq

def generate_events_at_pos_fast_z_fast_c3(pos_dict,numz,pos_num = 0,gap_frames = 3,z_offset = 0):
    """ *** should be general but I have not tested this for any other sequence besides the 3 channel fast imaging of feb-april 2023***
    Generates the list of image events for a single worm in 3 channels
    Channel 0: null channel for dumby images (1 at start, and # gap_frames between each channel for wheel motion time)
    Channel 1: fist channel, typically BF but doesnt matter here since computer is blind
    Channel 2: second channel
    channel 3: third channel

    each event has the axis {pos_number,c,z} for storing the set into the tiff when recieved
    only hardware instruction to mmcore is the xyz position
    (it used to be okay to just put the xyz for the first image since all are same but now it needs all of them)"""
    events = []
    #pos_num = 0
    null_z = 0 #init the starting z pos of the null

    #pos_dict =  pf.get_current_pos_dict(c,mode='xyz') #{'X':0,'Y':0, 'Z':0}
    #z_offset = 0

    # null pre frame
    events.append({"axes": {"p": pos_num, 'c':0, "z": null_z},
                   'x': pos_dict['X'], 'y': pos_dict['Y'], 'z': pos_dict['Z'] + z_offset})
    null_z+=1

    #channel 1
    for z in range(numz):
        events.append({"axes": {"p": pos_num, 'c':1, "z": z},'x': pos_dict['X'], 'y': pos_dict['Y'], 'z': pos_dict['Z'] + z_offset})
    # null gap frames between channels durring wheel motion and piezo motion down to start z pos
    for z in range(gap_frames):
        events.append({"axes": {"p": pos_num, 'c':0, "z": null_z},'x': pos_dict['X'], 'y': pos_dict['Y'], 'z': pos_dict['Z'] + z_offset})
        null_z+=1

    # channel 2
    for z in range(numz):
        events.append({"axes": {"p": pos_num, 'c':2, "z": z},'x': pos_dict['X'], 'y': pos_dict['Y'], 'z': pos_dict['Z'] + z_offset})

    # null gap frames between channels durring wheel motion and piezo motion down to start z pos
    for z in range(gap_frames):
        events.append({"axes": {"p": pos_num, 'c':0, "z": null_z},'x': pos_dict['X'], 'y': pos_dict['Y'], 'z': pos_dict['Z'] + z_offset})
        null_z+=1

    # channel 3
    for z in range(numz):
        events.append({"axes": {"p": pos_num, 'c':3, "z": z},'x': pos_dict['X'], 'y': pos_dict['Y'], 'z': pos_dict['Z'] + z_offset})
    return events

# def java_str_vector(bridge,str_array):
#     array_java = bridge._construct_java_object("mmcorej.StrVector")
#     for i in str_array:
#         array_java.add(str(i))
#     return array_java

# def flatL(ll):
#     lout = []
#     for l in ll:
#         lout.append(l[0])
#         lout.append(l[1])
#     return lout

def prep_wheel_seq(c):
    """ prep wheel for sequencing 3 channel at wheel positions 7,0,1 which corespond to blank,green,red,
    set speed to max, and go to the first position ready for triggers"""

    c.set_property('ASIFilterWheelSA','SpeedSetting','9')

    c.set_property('ASIFilterWheelSA','SerialCommand','P0 7')
    c.set_property('ASIFilterWheelSA','SerialCommand','P1 0')
    c.set_property('ASIFilterWheelSA','SerialCommand','P2 1')
    c.set_property('ASIFilterWheelSA','SerialCommand','P3 -1')
    c.set_property('ASIFilterWheelSA','SerialCommand','P4 -1')
    c.set_property('ASIFilterWheelSA','SerialCommand','P5 -1')
    c.set_property('ASIFilterWheelSA','SerialCommand','P6 -1')
    c.set_property('ASIFilterWheelSA','SerialCommand','P7 -1')

    c.set_property('ASIFilterWheelSA','SerialCommand','G0')
    return None

def load_ard_seq_multi_attempts(c,state_seq_java,attempts = 2):
    """for an unknow reason, uploading a long sequnce of trigger steps to the arduino, does not work on the first try,
    there is some error in communication. retry the same code does work though, not sure why. it might be a timeout error???

    here code will just try this 2 times with a time delay between attempts because there will be a worse error if you try the second time too fast"""
    attempts = 2
    for i in range(attempts):
        try:
            c.load_property_sequence('Arduino-Switch','State',state_seq_java)
            # If the communication was successful, you can break out of the loop
            print('successful load arduino queue')
            break
        except Exception as e:
            # If there was an error communicating with the hardware, print the error message
            print(f"Error communicating with hardware on attempt {i}: {e} ")
            if i == attempts - 1:
                # If this is the last attempt, raise an error
                raise RuntimeError("Hardware communication failed after multiple attempts")
            else:
                time.sleep(10)



def norm(input_list):
    min_value = min(input_list)
    max_value = max(input_list)
    normalized_list = [(x - min_value) / (max_value - min_value) for x in input_list]
    return normalized_list


def image_edge_array_xy_zStack(c,out_chan_ar,data_path,file_name = "TL_alignment_zstack",
                               reverse_ar = False, chan = 'TL',zrange = (-20,20),zstep = 2):
    """ Image zstack in brightfield at an array of positions.
    take in array of xyz and images a zstack at those positions, used zeissfocusaxis (nose) as the focus device,
    can spesify the zrange, zstep

    reverse_ar is bool param if you want to image in reverse order for efficiency, default False"""
    print("RUNNING IMAGING CODE")
    if reverse_ar:
        out_chan_ar.reverse()

    events = []
    for im, pos in enumerate(out_chan_ar):
        for j, z in enumerate(range(zrange[0], zrange[1], zstep)):
            events.append({'axes': {'im': im, 'z': j}, 'channel': {'group': 'LEDs_wheel_chan', 'config': chan},
                           'x': pos[0], 'y': pos[1], 'z': pos[2] + z})

    c.set_focus_device('ZeissFocusAxis')  # 'ZStage')
    c.set_config('LEDs_wheel_chan', chan)
    c.set_auto_shutter(True)

    #file_name = "TL_alignment_zstack_right_to_left"
    with Acquisition(directory=data_path, name=file_name, show_display=True, debug=False) as acq:
        acq.acquire(events)
        alignment_zstack_edge_dataset = acq.get_dataset()

    return alignment_zstack_edge_dataset



def image_edge_array_xy(c,a_edge_ar,data_path,file_name = "TL_pre_align_edges_pos"):
    """ takes in list of lists with xyz pos, creates an event list for imaging 1 bf image at each pos,
    prepares scope configes and carefully goes to the right pos,
    runs imaging and returns the dataset"""

    print("RUNNING IMAGING CODE")
    chan = "TL_noF"
    events = []
    for im, pos in enumerate(a_edge_ar):
        events.append({'axes': {'im': im}, 'channel': {'group': 'LEDs_wheel_chan', 'config': chan},
                       'x': pos[0], 'y': pos[1], 'z': pos[2]})
    go_to_xyz_careful(c, a_edge_ar[0], sleep=5)

    c.set_focus_device('ZeissFocusAxis')  # 'ZStage')
    c.set_config('LEDs_wheel_chan', chan)
    c.set_auto_shutter(True)

    #file_name = "TL_pre_align_edges_pos"
    with Acquisition(directory=data_path, name=file_name, show_display=True, debug=False) as acq:
        acq.acquire(events)
        pre_align_edge_dataset = acq.get_dataset()

    return pre_align_edge_dataset


def image_strip(c,b,strip, fname, display, zseq, start_z=0, numz=9,path = r"C:\Users\LevineLab\Documents\python notebooks"):
    """ard_seq must be already uploaded"""

    go_to_xyz_careful(c, strip.start_pos, sleep=7)
    c.start_property_sequence('Arduino-Switch', 'State')
    pos_sequence = zseq
    reset_piezo(c)

    c.set_focus_device('ZeissFocusAxis')
    c.set_auto_shutter(False)
    c.set_position('ZStage', start_z)
    upload_triangle_z_seq(c, b, pos_sequence)

    prep_wheel_seq(c)

    print('starting imaging strip {} at time: {}'.format(strip.name, datetime.datetime.ctime(datetime.datetime.now())))

    for d_num, pos_L in enumerate(strip.pos_LofL_s1):
        events = []
        pos_idx = 0
        for pn, pos in enumerate(pos_L):
            if pos['Used']:
                e = generate_events_at_pos_fast_z_fast_c3(pos, numz, pos_num=pos_idx, gap_frames=3)
                events += e
                pos_idx += 1

        file_name = fname + '_dev{}'.format(strip.device_names_list[d_num])  # "pre_exp_B{}".format(dev)
        with Acquisition(directory=path, name=file_name, show_display=display, debug=False) as acq:
            acq.acquire(events)

    c.stop_property_sequence('Arduino-Switch', 'State')
    reset_piezo(c)



def measure_sharpness_diag(im, cut_frac = 4, mediansize=3, filt=None, pool=True):
    """ not used anymore
    simple measure of image focus along the middle diagonal of the image. not very efficient code.
      used block reduce to reduce dimentions and smooth out noise
      then computes sobel like filter
      then sums and squares all pixels in the middle diagonal"""
    if pool:
        im = measure.block_reduce(im, (mediansize, mediansize), func=np.median)
    else:
        im = ndimage.median_filter(im, mediansize)
    if filt is None:
        filt = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    im = signal.convolve2d(im, filt, mode='same')
    s = 0
    rcut = int(im.shape[0] / cut_frac)
    for i in range(im.shape[0]):
        for j in range(im.shape[0]):
            if i > j - rcut and i < j + rcut:
                if i > rcut and i < im.shape[0] - rcut:
                    s += im[i, j] ** 2
    return s


def setup_20x_diag_tile_alignment(c, xyz_pos, strip_name,data_path,date,plot_figs = True):
    """takes a snap at the xyz_pos and 4 other snaps at shifted positions,
    then runs template match from the center snap to the other ones,
    then uses the image offset vectors and the know stage distances,
    it computes the rotation transformation matrix for the camera/stage.

    Params
    ------
    c -> core instance
    xyz_pos -> position tuple (x,y,z) where z if the zeiss focus axis
    strip_name -> string like 'A'
    data_path -> path for storing the images and data
    date -> date string
    plot_figs -> plot the process

    :returns
    -------
    center -> np array of the template
    tans_mat -> transformation matrix result

    """
    """ Set up positions and events of the shifted images """
    [x0, y0, z0] = xyz_pos
    dxy0 = [0, 0]
    dxy1 = [40, 30]
    dxy2 = [10, -20]
    dxy3 = [-20, 30]
    dxy4 = [-45, -35]
    dxys = [dxy0, dxy1, dxy2, dxy3, dxy4]

    if plot_figs:
        plt.figure()
        for dxy in dxys:
            plt.scatter(dxy[0], dxy[1]);
        plt.title("Relative pos of shifted images in xystage coords")
        plt.show()
        # pos_template_channel

    events = []
    for i, dxy in enumerate(dxys):
        events.append(
            {'axes': {'im': i}, 'channel': {'group': 'LEDs_wheel_chan', 'config': 'TL_noF'}, 'x': x0 + dxy[0], 'y': y0 + dxy[1],
             'z': z0})
    # events

    """ Acquire Images """
    c.set_focus_device('ZeissFocusAxis')  # 'ZStage')
    c.set_config('LEDs_wheel_chan', 'TL_noF')
    c.set_auto_shutter(True)

    file_name = "template_plus_minus_40xy".format(strip_name)
    with Acquisition(directory=data_path, name=file_name, show_display=True, debug=False) as acq:
        acq.acquire(events)
        template_dataset = acq.get_dataset()

    """ Get the the centered image and crop out the center"""
    template_stack = np.array(np.squeeze(template_dataset.as_array()))
    # template_stack.shape
    center = np.array(template_stack[0, 500:1700, 500:1700])

    #strip_obj.template_center = center
    if plot_figs:
        plt.figure()
        plt.imshow(center, 'gray')
        plt.title("Centered template, ideal pos")
        plt.show()

    """ Save and reopen center template """
    center_fname = 'center_{}_{}.npy'.format(strip_name, date)
    with open(os.path.join(data_path, center_fname), 'wb') as f:
        np.save(f, center)

    center_fname = 'center_{}_{}.npy'.format(strip_name, date)
    with open(os.path.join(data_path, center_fname), 'rb') as f:
        center = np.load(f)

    """ Template match find center position in the shifted images"""
    loc_err_ar = []
    for i in range(template_stack.shape[0]):
        loc_err_ar.append(af.find_chamber(template_stack[i, :, :], center))

    if plot_figs:
        """ Plot the centered tile overlaped on the shifted images to confirm template match is working correctly"""
        plt.figure(figsize=[20, 30])
        for i in range(template_stack.shape[0]):
            loc = np.array(loc_err_ar[i])
            im0 = template_stack[i, :, :]
            composite = im0.copy()
            # composite[loc[0]:loc[0]+center.shape[0],loc[1]:loc[1]+center.shape[1]]+=center
            composite[loc[1]:loc[1] + center.shape[1], loc[0]:loc[0] + center.shape[0]] += center
            plt.subplot(1, template_stack.shape[0], i + 1)
            plt.imshow(composite, 'gray')
            plt.title('stage shift {}->{}'.format(dxys[i], loc - 500))
        plt.show()


    # plt.savefig(os.path.join(path,'channel_pos_alignment.png'),format = 'png')

    def compute_rotation_trans(dxy,loc):
        dx = dxy[0]
        dy = dxy[1]

        xloc = loc[0] -500
        yloc = loc[1] -500
        a = np.array([[xloc, -yloc], [yloc, xloc]])
        b = np.array([dx, dy])
        x = np.linalg.solve(a, b)
        return  x #np.array([[x[, -yloc], [yloc, xloc]])

    """ Compute the cos(theta) and sin(theta) for the rotation matrix for each of the shifted images"""
    cs_til = []
    for i in range(1, len(dxys)):
        cs_til.append(compute_rotation_trans(dxys[i], loc_err_ar[i]))
    # cs_til

    """ Average the cos and sins between the images and compute and save the transformation matrix"""
    cs_til_avg = np.mean(np.array(cs_til), axis=0)
    # cs_til_avg
    trans_mat = np.array([[cs_til_avg[0], -cs_til_avg[1]], [cs_til_avg[1], cs_til_avg[0]]])
    # trans_mat
    trans_mat_fname = 'trans_mat_{}_{}.npy'.format(strip_name, date)
    with open(os.path.join(data_path, trans_mat_fname), 'wb') as f:
        np.save(f, trans_mat)

    """ Use trans mat to calculate the error from the known shifts"""
    errs = []
    for i in range(4):
        loc = loc_err_ar[i + 1]
        loc_v = np.array(loc) - 500
        # print(np.matmul(trans_mat,loc_v))
        errs.append(dxys[1:][i] - np.matmul(trans_mat, loc_v))
    # print(dxys[1:])
    print("pos prediction error in um:")
    print(np.array(errs))

    """ Show center template with the stage basis"""

    inv_trans_mat = np.linalg.inv(trans_mat)
    loc_100_0 = np.matmul(inv_trans_mat, [100, 0])
    loc_0_100 = np.matmul(inv_trans_mat, [0, 100])

    if plot_figs:
        plt.figure()
        plt.imshow(center, 'gray')
        c_center = [int(center.shape[0] / 2), int(center.shape[1] / 2)]
        plt.arrow(c_center[0], c_center[1], loc_100_0[0], loc_100_0[1], head_width=10)
        plt.arrow(c_center[0], c_center[1], loc_0_100[0], loc_0_100[1], head_width=10)
        plt.title('center template')
        plt.show()

    return center, trans_mat



##########################################################################################################
#
# the following is the class definition of a wormspa strip. this inlcudes 4 devices
# this is very useful to set up the imaging but durring the experiment its only used to access the
# positions of the worms and to know how to save the files
#
###########################################################################################################


class wspa_strip:
    def __init__(self, name, date, data_path, pump_num, description=''):
        self.name = name
        self.strip_name = name
        self.date = date
        self.data_path = data_path
        self.pump_num = pump_num
        self.description = ''
        self.template_center = None
        self.trans_mat = None
        self.template_tile_xyz = None
        self.excluded_pos = None
        self.device_names_list = None
        self.pos_per_dev = 32
        self.approx_edge_array = None
        self.xy_align_edge_ar = None
        self.out_chan_ar = None
        self.xyz_align_out_chan_ar = None
        self.bestz = None
        self.pos_LofL_s1 = None

    def set_nump_per_dev(self,num):
        """ default is 32 which is set at init but this can be changed"""
        self.pos_per_dev = num

    def set_exclude_pos(self,exclude_pos_l):
        self.excluded_pos = exclude_pos_l

    def set_device_names(self,names_list):
        self.device_names_list = names_list

    def run_tile_allignment(self,c,xyz_pos,plot_figs = True):
        """ calls the non class method 'setup_20x_diag_tile_alignment'
        uses self.name,datapath,date
        adds center and trans mat to the object attributes, and returns them
        """
        self.template_tile_xyz = xyz_pos
        self.template_center,self.trans_mat = setup_20x_diag_tile_alignment(c, xyz_pos,
                                                                            self.strip_name,
                                                                            self.data_path,
                                                                            self.date,plot_figs)
        return self.template_center,self.trans_mat

    def get_trans_mat_file(self,data_path = None,date = None,strip_name = None):
        """ get the center temlate and the transformation matrix from files in the data directory"""
        if data_path is None:
            data_path = self.data_path
        if date is None:
            date = self.date
        if strip_name is None:
            strip_name = self.strip_name
        center_fname = 'center_{}_{}.npy'.format(strip_name,date)
        with open(os.path.join(data_path, center_fname), 'rb') as f:
            center = np.load(f)

        trans_mat_fname = 'trans_mat_{}_{}.npy'.format(strip_name,date)
        with open(os.path.join(data_path, trans_mat_fname), 'rb') as f:
            trans_mat = np.load(f)
        self.template_center = center
        self.trans_mat = trans_mat
        return center,trans_mat

    def plot_stage_axis(self):
        """plot an overlay of the template image with arrows coresponding to the stage directions,
        used just for verification"""
        inv_trans_mat = np.linalg.inv(self.trans_mat)
        loc_100_0 = np.matmul(inv_trans_mat, [100, 0])
        loc_0_100 = np.matmul(inv_trans_mat, [0, 100])

        plt.figure()
        plt.imshow(self.template_center, 'gray')
        c_center = [int(self.template_center.shape[0] / 2), int(self.template_center.shape[1] / 2)]
        plt.arrow(c_center[0], c_center[1], loc_100_0[0], loc_100_0[1], head_width=10)
        plt.arrow(c_center[0], c_center[1], loc_0_100[0], loc_0_100[1], head_width=10)
        plt.title("Template tile, and XY Stage Basis")
        plt.show()
        return None

    def set_approx_edge_array(self,a_edge_ar):
        """ after manually adding the positions of the edges of the wspa devices for a strip,
        call this method to add them to the strip as approximate edges"""
        self.approx_edge_array = a_edge_ar

    def run_guided_xy_device_positioning(self,c,file_name = "TL_pre_align_edges_pos"):
        file_name = file_name+'{}_{}'.format(self.name,self.date)

        pre_align_edge_dataset = image_edge_array_xy(c,self.approx_edge_array,self.data_path,file_name)
        self.xy_align_edge_ar = adjust_edge_ar(pre_align_edge_dataset, self.template_center,
                                             self.approx_edge_array, self.trans_mat)

        # pf.plot_position_list(self.approx_edge_array)
        # pf.plot_position_list(self.xy_align_edge_ar, marker='+')
        # plt.legend('')
        # plt.title('')
        return self.xy_align_edge_ar

    def run_edge_ar_autoZ_measurement_RtoL(self,c,cut_size = 4,diff = 455.1,chan = 'TL'):
        """ legacy method march 2023"""
        """ no longer using this method.
        image the outer serpentine channel to get the most in focus z postition"""
        self.out_chan_ar = copy.deepcopy(self.xy_align_edge_ar)
        for i, edge in enumerate(self.out_chan_ar):
            if i % 2 == 0:
                edge[0] -= diff
            else:
                edge[0] += diff

        out_chan_ar_rev = copy.deepcopy(self.out_chan_ar)
        out_chan_ar_rev.reverse()
        alignment_zstack_edge_dataset = image_edge_array_xy_zStack(c, out_chan_ar_rev, self.data_path,
                                                                   file_name="TL_alignment_zstack_right_to_left_{}_{}".format(
                                                                       self.name,self.date),
                                                                   reverse_ar=False, chan=chan)
        print('image positions taken in reverse order: Right to Left')
        zdata = np.array(np.squeeze(alignment_zstack_edge_dataset.as_array()))
        #zdata.shape
        sharp_all_pos = []
        for pos in zdata:
            sharp = []
            for im in pos:
                sharp.append(measure_sharpness_diag(im,cut_size))
            sharp_all_pos.append(sharp)


        sharp_all_pos.reverse()

        self.bestz = [np.argmax(pos) for pos in sharp_all_pos]  # fixed back to Left -> Right

        sharp_ar = np.array(sharp_all_pos)
        peaks = [sharp_ar[i, self.bestz[i]] for i in range(len(self.bestz))]
        plt.figure()
        for sharp in sharp_all_pos:
            plt.plot(np.arange(20), sharp)
        plt.legend([i for i in range(len(sharp_all_pos))])
        plt.scatter(self.bestz, peaks, marker='o')
        plt.grid()
        plt.title('bestz: {}'.format(str(self.bestz)))

        self.xyz_align_out_chan_ar = copy.deepcopy(self.out_chan_ar)
        for i, pos in enumerate(self.xyz_align_out_chan_ar):
            pos[2] += list(range(-20, 20, 2))[self.bestz[i]]
        z = [p[2] for p in self.xyz_align_out_chan_ar]
        x = [p[0] for p in self.xyz_align_out_chan_ar]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, z)
        plt.figure()
        plt.plot(x, z, marker='*')
        zl = np.array(x) * slope + intercept
        plt.plot(x, zl)
        plt.legend(['autof_edges', 'linear_fit_zline'])

        self.xyz_adj_edge_ar = copy.deepcopy(self.xy_align_edge_ar)
        z = [p[2] for p in self.xyz_align_out_chan_ar]
        x = [p[0] for p in self.xyz_align_out_chan_ar]
        for j, i in enumerate(range(0, len(self.xyz_align_out_chan_ar), 2)):
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[i:i + 2], z[i:i + 2])
            self.xyz_adj_edge_ar[i][2] = np.array(self.xyz_adj_edge_ar[i][0]) * slope + intercept
            self.xyz_adj_edge_ar[i + 1][2] = np.array(self.xyz_adj_edge_ar[i + 1][0]) * slope + intercept

        return self.bestz,sharp_all_pos,self.xyz_adj_edge_ar

    def edit_bestz_and_adjust_edge_ar(self,new_bestz,sharp_all_pos = None):
        """ legacy version march 2023, uses sobel for sharpness measure... no longer used
        not ideal because channel depth is larger than the objective dept of field"""

        #
        #
        # needs fixing the reverse stuff
        #
        #
        #
        """"If imaging the outer chan has already been imaged, then use this to readjust the bestz pos for a tile
        and then recompute the corrected outer chan ar and the xyz_adjusted_edge_ar

        so essentially same as the run_edge_ar_autoZ_measurement_RtoL but without the round of imaging."""
        if self.bestz is None:
            print('best z is non, need to image first')
            return None

        self.bestz = new_bestz

        if sharp_all_pos is not None:
            # plt.figure()
            # for sharp in sharp_all_pos:
            #     plt.plot(np.arange(20), sharp)
            # plt.legend([i for i in range(len(sharp_all_pos))])
            # plt.scatter(np.arange(len(self.bestz)), self.bestz, marker='o')
            # plt.grid()
            # plt.title('bestz: {}'.format(str(self.bestz)))
            #sharp_all_pos.reverse()

            sharp_ar = np.array(sharp_all_pos)
            peaks = [sharp_ar[i, self.bestz[i]] for i in range(len(self.bestz))]
            plt.figure()
            for sharp in sharp_all_pos:
                plt.plot(np.arange(20), sharp)
            plt.legend([i for i in range(len(sharp_all_pos))])
            plt.scatter(self.bestz, peaks, marker='o')
            plt.grid()
            plt.title('bestz: {}'.format(str(self.bestz)))

        self.xyz_align_out_chan_ar = copy.deepcopy(self.out_chan_ar)
        for i, pos in enumerate(self.xyz_align_out_chan_ar):
            pos[2] += list(range(-20, 20, 2))[self.bestz[i]]
        z = [p[2] for p in self.xyz_align_out_chan_ar]
        x = [p[0] for p in self.xyz_align_out_chan_ar]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, z)
        plt.figure()
        plt.plot(x, z, marker='*')
        zl = np.array(x) * slope + intercept
        plt.plot(x, zl)
        plt.legend(['autof_edges', 'linear_fit_zline'])

        self.xyz_adj_edge_ar = copy.deepcopy(self.xy_align_edge_ar)
        z = [p[2] for p in self.xyz_align_out_chan_ar]
        x = [p[0] for p in self.xyz_align_out_chan_ar]
        for j, i in enumerate(range(0, len(self.xyz_align_out_chan_ar), 2)):
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[i:i + 2], z[i:i + 2])
            self.xyz_adj_edge_ar[i][2] = np.array(self.xyz_adj_edge_ar[i][0]) * slope + intercept
            self.xyz_adj_edge_ar[i + 1][2] = np.array(self.xyz_adj_edge_ar[i + 1][0]) * slope + intercept

        return self.bestz, sharp_all_pos, self.xyz_adj_edge_ar

    def generate_pos_list_for_strip(self):
        '''requires self to have:
        xyz_adjusted_edge_array (xyz_adj_edge_ar),
        list of positions to exclude,
        number of positions per device
        and names,

        generates a list of lists (1 list for each device), the inner list has dictionaries at each of the positions which denote:
         the position number, the device name, the x,y,z values, and a bool for if it should be imaged

         also sets the starting position of the strip'''
        self.pos_LofL_s1 = pos_list_whole_strip(self.xyz_adj_edge_ar, exclude_LofL=self.excluded_pos, dev_tiles=self.pos_per_dev,
                                              dev_names_list=self.device_names_list)
        self.start_pos = self.xyz_adj_edge_ar[0]
        return self.pos_LofL_s1

    def save_edge_ar(self):
        edge_fname = 'edge_ar_final_{}_{}.npy'.format(self.name,self.date)
        with open(os.path.join(self.data_path, edge_fname), 'wb') as f:
            np.save(f, np.array(self.xy_align_edge_ar))
        return None

    def save_pos_lofl(self):
        """not implemented"""
        return None

    def get_attribute_dict(self):
        return vars(self)

    def save_strip_dict(self,strip_dict, f_name):
        out_dict = copy.deepcopy(strip_dict)
        for key, value in out_dict.items():
            if isinstance(value, np.ndarray):
                out_dict[key] = value.tolist()
            if value is None:
                out_dict[key] = 'null'
        with open(f_name, 'w') as f:
            json.dump(out_dict, f)
    def pickle_save(self,f_name):
        with open(f_name, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def open_pickled_obj(cls,file_path):
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def run_guided_xyz_autoZ_positioning(self,c,zrange = [-16,40],z_step=2,file_name = None,data_path = None,plot_figs = True,auto_accept = True):
        """ current zstack allignment method as of 4/20"""
        if file_name is None:
            file_name = 'strip {} zfocusing edges'.format(self.strip_name)
        if data_path is None:
            data_path = self.data_path
        z_data_allpos = image_multi_focusing_zstack(c,
                                                   self.xy_align_edge_ar, zrange, z_step,
                                                   file_name=file_name,
                                                   data_path=data_path)
        zdata_ar = np.squeeze(z_data_allpos.as_array())
        print('zdata shape:',zdata_ar.shape)

        # compute radial mean of power spectrum of shifted FFT for all images in all position zstacks
        self.rm = np.array([np.array([radial_mean(measure.block_reduce(np.array(image), (3, 3), func=np.median))
                                      for image in stack]) for stack in zdata_ar])
        # sum the radial mean over the arbitrary frequency range that has been found to have a peak at the bottom of the channel
        self.sum_rm = np.sum(self.rm[:, :, 200:-150], axis=2)
        #self.sum_rm.shape
        self.suf_idx_zstack = [find_increasing_values(norm(s))[1] for s in self.sum_rm]
        if plot_figs:
            plt.figure()
            for i, s in enumerate(self.sum_rm):
                plt.plot(np.arange(len(s)), norm(s))
            plt.legend(['{}'.format(j) for j in range(self.sum_rm.shape[0])])

            plt.figure(figsize=[40, 5])
            for i in range(self.sum_rm.shape[0]):
                plt.subplot(1, self.sum_rm.shape[0], i + 1)
                ydata = norm(self.sum_rm[i])
                z_positions = np.arange(0, len(ydata))
                s_idx = 0
                e_idx = len(ydata)
                p0 = [2, 10]
                # popt,pcov = curve_fit(logistic_l,z_positions[s_idx:e_idx],ydata,p0)
                plt.plot(z_positions[s_idx:e_idx], ydata)
                plt.title('max at: {}, chosen: {}'.format(np.argmax(ydata),self.suf_idx_zstack[i]))
                #_, inc_idx = pf.find_increasing_values(ydata)
                plt.axvline(self.suf_idx_zstack[i], color='green')
        if auto_accept:
            xy_align_edges = copy.deepcopy(self.xy_align_edge_ar)
            #zstep = 2
            #zrange = [-16, 40]
            zstack_pos = np.arange(zrange[0], zrange[1], z_step)
            for i, pos in enumerate(xy_align_edges):
                pos[2] += zstack_pos[self.suf_idx_zstack[i]]

            self.xyz_adj_edge_ar = xy_align_edges

    def compute_zpositioning_from_file(self,dataset_path, zrange = [-16,40],z_step=2,plot_figs = True,auto_accept = True):
        zdata_ar = np.squeeze(Dataset(dataset_path).as_array()).transpose(1, 0, 2, 3)
        print(zdata_ar.shape)

        # compute radial mean of power spectrum of shifted FFT for all images in all position zstacks
        self.rm = np.array([np.array([radial_mean(measure.block_reduce(np.array(image), (3, 3), func=np.median))
                                      for image in stack]) for stack in zdata_ar])
        # sum the radial mean over the arbitrary frequency range that has been found to have a peak at the bottom of the channel
        self.sum_rm = np.sum(self.rm[:, :, 200:-150], axis=2)
        #self.sum_rm.shape
        self.suf_idx_zstack = [find_increasing_values(norm(s))[1] for s in self.sum_rm]
        if plot_figs:
            plt.figure()
            for i, s in enumerate(self.sum_rm):
                plt.plot(np.arange(len(s)), norm(s))
            plt.legend(['{}'.format(j) for j in range(self.sum_rm.shape[0])])

            plt.figure(figsize=[40, 5])
            for i in range(self.sum_rm.shape[0]):
                plt.subplot(1, self.sum_rm.shape[0], i + 1)
                ydata = norm(self.sum_rm[i])
                z_positions = np.arange(0, len(ydata))
                s_idx = 0
                e_idx = len(ydata)
                p0 = [2, 10]
                # popt,pcov = curve_fit(logistic_l,z_positions[s_idx:e_idx],ydata,p0)
                plt.plot(z_positions[s_idx:e_idx], ydata)
                plt.title('max at: {}, chosen: {}'.format(np.argmax(ydata),self.suf_idx_zstack[i]))
                #_, inc_idx = pf.find_increasing_values(ydata)
                plt.axvline(self.suf_idx_zstack[i], color='green')
        if auto_accept:
            xy_align_edges = copy.deepcopy(self.xy_align_edge_ar)

            zstack_pos = np.arange(zrange[0], zrange[1], z_step)
            for i, pos in enumerate(xy_align_edges):
                pos[2] += zstack_pos[self.suf_idx_zstack[i]]

            self.xyz_adj_edge_ar = xy_align_edges





############################################################################################################
#
# the following is code for creating a pumps program
#
############################################################################################################
def compute_steps_time(steps):
    runtime = 0
    for s in steps:
        if s['type']=='pvflow':
            runtime+=(s['v']/s['r'])+s['post_wait']/60
    return runtime

def get_steps_1port_inject_flow(source = 7,chip = 2,waste = 3,buffer = 4):

    steps = [{'type':'pvflow','p':source,'r':300,'v':300,'d':'Withdraw','post_wait':4,'status':'1port injection flow, step1/7, withdraw p{}'.format(source)},
             {'type':'pvflow','p':chip,'r':200,'v':100,'d':'Infuse','post_wait':4,'status':'1port injection flow, step2/7 pulse to device p{}'.format(chip)},
             {'type':'pvflow','p':chip,'r':12,'v':48,'d':'Infuse','post_wait':4,'status':'1port injection flow, step3/7 flow to device p{}'.format(chip)},
             {'type':'pvflow','p':chip,'r':200,'v':100,'d':'Infuse','post_wait':4,'status':'1port injection flow, step4/7 pulse to device p{}'.format(chip)},
             {'type':'pvflow','p':chip,'r':12,'v':48,'d':'Infuse','post_wait':4,'status':'1port injection flow, step5/7 flow to device p{}'.format(chip)},
             {'type':'pvflow','p':waste,'r':600,'v':300,'d':'Infuse','post_wait':4,'status':'1port injection flow, step6/7 to waste p{}'.format(waste)},
             {'type':'pvflow','p':buffer,'r':600,'v':300,'d':'Withdraw','post_wait':4,'status':'1port injection flow, step7/7 wdr buffer p{}'.format(buffer)}]
    return steps

def get_steps_wash_chip(chip = 2,waste = 3,buffer = 4):
    steps = [{'type':'pvflow','p':chip,'r':200,'v':300,'d':'Infuse','post_wait':4,'status':'wash chip,step1/3, pulse buffer to chip p{}'.format(chip)},
             {'type':'pvflow','p':chip,'r':12,'v':72,'d':'Infuse','post_wait':4,'status':'wash chip,step2/3, flow buffer to chip p{}'.format(chip)},
             {'type':'pvflow','p':buffer,'r':600,'v':372,'d':'Withdraw','post_wait':4,'status':'wash chip,step3/3, refill buffer p{}'.format(buffer)}]
    return steps


def get_steps_flow_simpletest(source = 7,chip = 2,waste = 3,buffer = 4):

    steps = [{'type':'pvflow','p':source,'r':1000,'v':300,'d':'Withdraw','post_wait':4,'status':'test,step1 withdraw p{}'.format(source)},
             {'type':'pvflow','p':chip,'r':200,'v':10,'d':'Infuse','post_wait':4,'status':'test,step2 p{}'.format(chip)}]
    return steps

def get_steps_switch_source(source = 8,chip = 2,waste = 3,buffer = 4):

    steps = [{'type':'pvflow','p':source,'r':300,'v':300,'d':'Withdraw','post_wait':4,'status':'clean source, step1/3, withdraw p{}'.format(source)},
             {'type':'pvflow','p':waste,'r':300,'v':600,'d':'Infuse','post_wait':4,'status':'clean source, step2/3 to waste p{}'.format(waste)},
             {'type':'pvflow','p':buffer,'r':600,'v':300,'d':'Withdraw','post_wait':4,'status':'clean source, step 3/3 wdr buffer p{}'.format(buffer)}]
    return steps

def get_steps_notify(message,fname = 'jupyter_com'):
    return [{'type':'notify','com_file':r'C:\Users\LevineLab\Documents\Repos\Pumps\{}'.format(fname),
             'message':'{}'.format(message)}]

def get_steps_make_schedule(cycle_times):
    return [{'type':'make_schedule','file_name':r'C:\Users\LevineLab\Documents\Repos\Pumps\wash_schedule','cycle_times':cycle_times}]

def get_approx_times_between_notify(steps):
    cycle_times = []
    runtime = 0
    for s in steps:
        if s['type']=='pvflow':
            runtime+=(s['v']/s['r'])+s['post_wait']/60
        elif s['type']=='notify':
            cycle_times.append(round(runtime,2))
            runtime = 0
    return cycle_times


def create_PV_program_sequential_2_species(source1_port = 7,source2_port = 8,source1_cycles = 5,source2_cycles = 5):
    master_steps = []
    # pump_num = 0
    """ injection flow cycles from source 1, cycles are about 11.5 minutes"""
    for cycle in range(source1_cycles):
        for inject in range(5):
            master_steps += get_steps_1port_inject_flow(source=source1_port, chip=2, waste=3, buffer=4)

        master_steps += get_steps_notify('starting_wash {}'.format(cycle), fname='jupyter_com_pumps')
        master_steps += get_steps_wash_chip(chip=2, waste=3, buffer=4)

    """ when switching source, dispense to waste a full ml to clear out remaining species 1,
    and withdraw from source and dispense to waste to re prime source 2 tubing"""
    master_steps += get_steps_switch_source(source=source2_port, chip=2, waste=3, buffer=4)

    """ injection flow cycles from source 2, cycles are about 11.5 minutes"""
    for cycle in range(source1_cycles, source1_cycles+source2_cycles):
        for inject in range(5):
            master_steps += get_steps_1port_inject_flow(source=source2_port, chip=2, waste=3, buffer=4)

        master_steps += get_steps_notify('starting_wash {}'.format(cycle), fname='jupyter_com_pumps')

        master_steps += get_steps_wash_chip(chip=2, waste=3, buffer=4)
    """compute the approximate time for each cycle and add this list as a make schedule step"""
    cycle_time = get_approx_times_between_notify(master_steps)
    master_steps = get_steps_make_schedule(cycle_time) + master_steps
    return master_steps



def write_pump_program_json(master_steps,file_name=None):
    """write the sequnce PV steps into a json
    file will have 'name'=filename, and 'sequence_steps'=master_steps
    witten in the directory of the pumping program
    """
    if file_name is None:
        file_name = "pump_prog_4_26_23_1.json"
    pump_prog = {'name': file_name,
                 'sequence_steps': master_steps}
    with open(r'C:\Users\LevineLab\Documents\Repos\Pumps\{}'.format(pump_prog['name']), 'w') as f:
        json.dump(pump_prog, f, indent=4)


def wait_for_wash_notif(pump_num,wash_num):
    step_num = wash_num  # The step number you're waiting for
    time_limit = datetime.timedelta(seconds=90)
    #filename = 'notification.txt'
    #pump_num = 0
    filename = r'C:\Users\LevineLab\Documents\Repos\Pumps\jupyter_com_pumps_{}'.format(pump_num)

    def parse_notification(line):
        parts = line.strip().split()
        timestamp = datetime.datetime.fromisoformat(parts[0]+" "+parts[1])
        #name = parts[1]
        step = int(parts[-1])

        return timestamp, step

    found_notif = False
    while not found_notif:
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            timestamp, step = parse_notification(line)
            print(step)

            if step == step_num:
                time_elapsed = datetime.datetime.now() - timestamp
                remaining_time = time_limit - time_elapsed
                found_notif = True
                print("found line")
                if remaining_time > datetime.timedelta(0):
                    print('waiting..',remaining_time.total_seconds())
                    time.sleep(remaining_time.total_seconds())

                # Execute the code you want to run after the notification
                print("Received notification from script1. Continuing...")
                break
        else:
            print('sleeping')
            time.sleep(30)


###########################################################################################################
#
#
# the following is the class definitions for a Call and for ExpQUE
#
# this code is not quite ready for deployment
################################################################################################################

class Call:
    def __init__(self, name, expiry_time, func, args):
        self.name = name
        self.expiry_time = expiry_time
        self.func = func
        self.args = args

    def __lt__(self, other):
        return self.expiry_time < other.expiry_time


class Queue:
    def __init__(self):
        self.heap = []
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.thread = None
        self.paused = False

    # def get_heap(self):
    #     return self.heap

    def log(self, message):
        print(f"[{datetime.datetime.now()}] {message}")

    def add_call(self, call):
        with self.lock:
            heapq.heappush(self.heap, call)
            self.cv.notify()
            self.log("adding call {},for time {}".format(call.name, call.expiry_time))

    def remove_call(self, call):
        with self.lock:
            self.heap.remove(call)
            heapq.heapify(self.heap)
            self.cv.notify()
            self.log("removing call")

    def edit_call(self, call, new_call):
        """ not tested """
        with self.lock:
            self.remove_call(call)
            self.add_call(new_call)
            self.log("editing call")

    def get_heap(self):
        with self.lock:
            return heapq.nsmallest(len(self.heap), self.heap)

    def run(self):
        while True:
            with self.lock:
                if not self.heap:
                    self.log("Queue is empty. Exiting...")
                    break
                next_timep = self.heap[0].expiry_time
                wait_time = max(0, next_timep - datetime.datetime.now())
                if self.paused:
                    print('paused, waiting...')
                    self.cv.wait()
                    print('resumed')
                elif wait_time == 0:
                    next_call = heapq.heappop(self.heap)
                    self.log("Executing call to {} with arguments {}".format(next_call.func.__name__, next_call.name))
                    next_call.func(*next_call.args)
                else:
                    self.log("Waiting until {} for next call".format(next_timep))
                    self.cv.wait(wait_time)

    def start(self):
        with self.lock:
            if not self.thread:
                self.thread = threading.Thread(target=self.run)
                self.thread.start()
                self.log("Queue started")

    def stop(self):
        with self.lock:
            self.thread = None
            self.paused = False
            self.cv.notify()
            self.log("Queue stopped")

    def pause(self):
        with self.lock:
            self.paused = True
            self.cv.notify()
            self.log("Queue paused")

    def resume(self):
        with self.lock:
            self.paused = False
            self.cv.notify()
            self.log("Queue Resumed")