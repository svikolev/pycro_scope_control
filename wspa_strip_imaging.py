import threading
import datetime
from . import pycro_funcs as pf
import heapq

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import time


import os
import shutil


def create_call_set_wait_imaging_1strip(c, b, call_set_params):
    call_list = []
    for i, dt in enumerate(call_set_params['dt_list']):
        c_name = 'Image_Strip{}_num{}'.format(call_set_params['strip_obj'].name, call_set_params['wash_nums'][i])
        # args for imaging function

        fname = 'Strip{}_{}_{}h'.format(call_set_params['strip_obj'].name, call_set_params['exp_name'],
                                        call_set_params['wash_nums'][i])

        args = [{'c_name': c_name,
                 'fname': fname,
                 'pump_ID': call_set_params['pump_ID'],
                 'wash_num': call_set_params['wash_nums'][i],
                 'core': c, 'bridge': b,
                 'strip_obj': call_set_params['strip_obj'],
                 'display': call_set_params['display'],
                 'zseq': call_set_params['zseq'],
                 'start_z': call_set_params['start_z'],
                 'numz': call_set_params['numz'],
                 'path': call_set_params['strip_obj'].data_path}]
        call = Call(c_name, dt, call_set_params['func_name'], args)
        call_list.append(call)
    return call_list


def create_call_set_noWait_imaging_1strip(c, b, call_set_params):
    call_list = []
    for i, dt in enumerate(call_set_params['dt_list']):
        if 'suffix' in call_set_params:
            suffix = call_set_params['suffix']
        else:
            suffix = ''
        c_name = 'Image_Strip{}_num{}_{}'.format(call_set_params['strip_obj'].name, call_set_params['wash_nums'][i],
                                                 call_set_params['suffix'])
        # args for imaging function

        fname = 'Strip{}_{}_{}h{}'.format(call_set_params['strip_obj'].name, call_set_params['exp_name'],
                                          call_set_params['wash_nums'][i], call_set_params['suffix'])

        args = [{'c_name': c_name,
                 'fname': fname,
                 'pump_ID': call_set_params['pump_ID'],
                 'wash_num': call_set_params['wash_nums'][i],
                 'core': c, 'bridge': b,
                 'strip_obj': call_set_params['strip_obj'],
                 'display': call_set_params['display'],
                 'zseq': call_set_params['zseq'],
                 'start_z': call_set_params['start_z'],
                 'numz': call_set_params['numz'],
                 'path': call_set_params['strip_obj'].data_path}]
        call = Call(c_name, dt, call_set_params['func_name'], args)
        call_list.append(call)
    return call_list


def waitForWash_imageStrip_single_v2(args):
    c = args['core']
    b = args['bridge']

    cur_pos = pf.get_current_pos_dict(c, mode='xyz')
    strip = args['strip_obj']

    # check if scope is on current strip by seeing if the delta of current y and start y is greater then 3 mm
    # if so we need to move to safe z before moving to pos
    if np.abs(cur_pos['Y'] - strip.start_pos[1]) > 3000:
        strip.go_to_safez(c)
        time.sleep(2)
        pf.go_to_xyz_careful(c, strip.start_pos, sleep=7)

    # wait for the pump to signal that it is washing strip
    wait_for_wash_notif_v2(args['pump_ID'], args['wash_num'])

    pf.image_strip(c, b, strip, args['fname'], args['display'], zseq=args['zseq'], start_z=args['start_z'],
                   numz=args['numz'], path=args['path'])


def waitForWash_imageStrip_double_v2(args):
    c = args['core']
    b = args['bridge']

    cur_pos = pf.get_current_pos_dict(c, mode='xyz')
    strip = args['strip_obj']

    # check if scope is on current strip by seeing if the delta of current y and start y is greater then 3 mm
    # if so we need to move to safe z before moving to pos
    if np.abs(cur_pos['Y'] - strip.start_pos[1]) > 3000:
        strip.go_to_safez(c)
        time.sleep(2)
        pf.go_to_xyz_careful(c, strip.start_pos, sleep=7)

    # wait for the pump to signal that it is washing strip
    wait_for_wash_notif_v2(args['pump_ID'], args['wash_num'])

    print('imaging round 1...')
    pf.image_strip(c, b, strip, args['fname'] + "r1", args['display'], zseq=args['zseq'], start_z=args['start_z'],
                   numz=args['numz'], path=args['path'])

    print('imaging round 2...')
    pf.image_strip(c, b, strip, args['fname'] + "r2", args['display'], zseq=args['zseq'], start_z=args['start_z'],
                   numz=args['numz'], path=args['path'])


def noWait_imageStrip_double(args):
    c = args['core']
    b = args['bridge']

    cur_pos = pf.get_current_pos_dict(c, mode='xyz')
    strip = args['strip_obj']

    # check if scope is on current strip by seeing if the delta of current y and start y is greater then 3 mm
    # if so we need to move to safe z before moving to pos
    if np.abs(cur_pos['Y'] - strip.start_pos[1]) > 3000:
        strip.go_to_safez(c)
        time.sleep(2)
        pf.go_to_xyz_careful(c, strip.start_pos, sleep=7)
    print('imaging round 1...')
    pf.image_strip(c, b, strip, args['fname'] + "r1", args['display'], zseq=args['zseq'], start_z=args['start_z'],
                   numz=args['numz'], path=args['path'])
    print('imaging round 2...')
    pf.image_strip(c, b, strip, args['fname'] + "r2", args['display'], zseq=args['zseq'], start_z=args['start_z'],
                   numz=args['numz'], path=args['path'])


def noWait_imageStrip_single(args):
    c = args['core']
    b = args['bridge']

    cur_pos = pf.get_current_pos_dict(c, mode='xyz')
    strip = args['strip_obj']

    # check if scope is on current strip by seeing if the delta of current y and start y is greater then 3 mm
    # if so we need to move to safe z before moving to pos
    if np.abs(cur_pos['Y'] - strip.start_pos[1]) > 3000:
        strip.go_to_safez(c)
        time.sleep(2)
        pf.go_to_xyz_careful(c, strip.start_pos, sleep=7)

    pf.image_strip(c, b, strip, args['fname'] + "r1", args['display'], zseq=args['zseq'], start_z=args['start_z'],
                   numz=args['numz'], path=args['path'])


def read_datetime_file(file_path):
    """for opening the wash schedule created by pump when the pump program is started"""
    datetime_list = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            date_time_obj = datetime.datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f')
            datetime_list.append(date_time_obj)

    return datetime_list


def wait_for_wash_notif_v2(pump_num, wash_num, w_time=90):
    """ for reading the notif file from pumps. every 30s it will read the file and look for a line:
    'starting_wash i' where i is the wash number
    if it has found this line, then it will check the date time it was written, it will set a remaining time of 90s
    from that point and wait untill then."""
    step_num = wash_num  # The step number you're waiting for
    time_limit = datetime.timedelta(seconds=w_time)
    # filename = 'notification.txt'
    # pump_num = 0
    filename = r'C:\Users\LevineLab\Documents\Repos\PumpValveSystem\pump_coms\jupyter_com_pumps_{}.txt'.format(pump_num)

    def parse_notification(line):
        parts = line.strip().split()
        timestamp = datetime.datetime.fromisoformat(parts[0] + " " + parts[1])
        # name = parts[1]
        step = int(parts[-1])

        return timestamp, step

    found_notif = False
    while not found_notif:
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            timestamp, step = parse_notification(line)
            # print(step)

            if step == step_num:
                time_elapsed = datetime.datetime.now() - timestamp
                remaining_time = time_limit - time_elapsed
                found_notif = True
                print("found line")
                if remaining_time > datetime.timedelta(0):
                    print('waiting..', remaining_time.total_seconds())
                    time.sleep(remaining_time.total_seconds())

                # Execute the code you want to run after the notification
                print("Received notification from script1. Continuing...")
                break
        else:
            # print('sleeping')
            time.sleep(30)


def edit_timelist_add_X_minutes(datetimes, mins=20):
    result = []
    delta = datetime.timedelta(minutes=mins)

    for dt in datetimes:
        new_dt = dt + delta
        result.append(new_dt)

    return result



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
        while self.thread is not None:  # gpt4 says this is a bad approach. better to regularly check if should_continue so it can properly clean up
            with self.lock:
                if not self.heap:
                    self.log("Queue is empty. Exiting...")
                    break
                next_timep = self.heap[0].expiry_time
                wait_time = max(0, (next_timep - datetime.datetime.now()).total_seconds())
                print('wt', wait_time)
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





######################################################
#
# code for setting up overnight imaging queue setup
#
############################################################


def get_recently_modified_folders(directory, top_n=2):
    """Get the top_n most recently modified folders from the directory."""
    # List all folders in the directory
    all_folders = [os.path.join(directory, d) for d in os.listdir(directory) if
                   os.path.isdir(os.path.join(directory, d))]

    # Sort the folders by modification time in descending order
    sorted_folders = sorted(all_folders, key=lambda x: os.path.getmtime(x), reverse=True)

    return sorted_folders[:top_n]


def transfer_folders(src_directory, dest_directory, top_n=2):
    """Transfer the top_n most recently modified folders from src_directory to dest_directory."""
    # Ensure destination directory exists
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Get the top_n most recently modified folders from src_directory
    folders_to_transfer = get_recently_modified_folders(src_directory, top_n)

    # Transfer each folder
    for folder in folders_to_transfer:
        folder_name = os.path.basename(folder)
        dest_folder = os.path.join(dest_directory, folder_name)

        # Move the folder
        shutil.move(folder, dest_folder)
        print(f"Moved {folder} to {dest_folder}")

def move_files(numf=1, src_dir=r"X:\Lion_SK\Wspa_EXPERIMENT_11_10_23",
    dest_dir = r"L:\sk_lynx\Wspa_EXPERIMENT_11_10_23"):
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)
    transfer_folders(src_dir, dest_dir, numf)