# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import analysis_funcs as af
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    path = r"C:\Users\LevineLab\Documents\python notebooks\9_21_22_Wspa"
    fname_base = "BF_505_20ms_10z"

    d = 1  # device names are 1,2,3,4
    t = 0  # times are every hour starting with 0-9
    sufix = ''  # either empty or has 'nowash_'

    dataset, device_meta = af.open_device_dataset(path, fname_base, device=d, time=t, sufix=sufix)

    center_fname = 'center_9_22_22.npy'
    with open(os.path.join(path,center_fname), 'rb') as f:
        center = np.load(f)

    plt.imshow(center)
    plt.show()

