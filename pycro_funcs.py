import numpy as np
import time


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


def upload_triangle_z_seq(core, bridge, pos_sequence):
    """sends sequence to piezo stage
    needs core, bridge, and pos_sequence list
    """
    z_stage = 'ZStage'
    # core.get_focus_device() # just a string of device name, will be 'ZStage'
    # create java obj with the seq
    pos_seq_java = bridge._construct_java_object("mmcorej.DoubleVector")

    for i in pos_sequence:
        pos_seq_java.add(float(i))

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


def get_pos_list_from_manager(studio):
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

def set_pre_acq_params(core,exposure_t = 30):
    """ acquisition params"""

    # set up camera config ###
    core.set_config('Ham_Triggers', 'Out_Exp_Pos')
    core.wait_for_config('Ham_Triggers', 'Out_Exp_Pos')
    core.set_exposure(exposure_t)
    # set up camera config ###

    # set up TL config ###
    # core.set_config('TL_lamp', 'BF_2_5')
    # core.wait_for_config('TL_lamp', 'BF_2_5')
    core.set_config('TL_lamp', 'BF_20x_25v_55na_5bf')
    core.wait_for_config('TL_lamp', 'BF_20x_25v_55na_5bf')
    # set up TL config ###

    # set up cube config ###
    core.set_config('Cubes', 'semrock_TBP')
    core.wait_for_config('Cubes', 'semrock_TBP')
    # set up cube config ###

    # set up Light Path config ###
    core.set_config('Light_Path', 'Cam')
    core.wait_for_config('Light_Path', 'Cam')
    # set up Light Path config ###

    core.set_channel_group('Channel')
    core.set_position('ZStage', 0)
    ### check colibri shutter is main shutter and remains open, and autoshutter is off##

    core.set_shutter_device('ZeissColibri')
    core.set_auto_shutter(False)

def check_acq_configs():
    """need to add this later"""
    print("testing reImport")

def events_TL_multi_pos(pos_list,offset = 0):
    """ takes list of pos dictionaies and creates a list of TL events
    the axis is just """
    #pos_list = B1
    events = []
    im_num = 0
    for pos_num, pos_dict in enumerate(pos_list):
        if pos_dict['Used']:
            evt = { 'axes': {'im':im_num},
                   'channel': {'group': 'Channel', 'config': 'TL'},
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


def reset_piezo(core):
    core.set_property('ZStage', "Use Fast Sequence", "No")
    # core.set_property('ZStage', "Use Sequence", "No")
    core.set_position('ZStage', 0)
    return core.get_position('ZStage')

