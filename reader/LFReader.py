import file_io
import h5py
import numpy as np
import os
import glob
import cv2
import pdb
import imageio

def read_hci(scene_path, mask=False):
    f = h5py.File(scene_path, 'r')
    LF = f['LF']
    LF = LF[:]
    LF = LF[:, ::-1].copy()

    depth = f['GT_DEPTH'][4, 4]
    dH = f.attrs['dH']
    fo = f.attrs['focalLength']
    dx = f.attrs['shift']
    disparity = dH * fo / depth - dx

    if mask:
        mask = f['GT_DEPTH_MASK']
        mask = mask[4, 4]
        mask = mask[:] / 255.0

        return LF, disparity, mask

    return LF, disparity


def read_newhci(scene_path, read_disparity=False):
    img_list = sorted(glob.glob(os.path.join(scene_path, '*.png')))
    LF = np.zeros((9, 9, 512, 512, 3), dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            LF[i, j] = imageio.imread(img_list[i*9 + j])

    if read_disparity:
        disparity = file_io.read_disparity(scene_path, highres=False).copy()
    else:
        disparity = 0

    return LF, disparity

def read_inria(scene_path):
    img_list = sorted(glob.glob(os.path.join(scene_path, '*.png')))
    LF = np.zeros((9, 9, 512, 512, 3), dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            LF[i, j] = cv2.imread(img_list[i*9 + j])

    disparity = np.load(os.path.join(scene_path, 'disparity_5_5.npy'))

    return LF, disparity

if __name__ == '__main__':
    scene_path = '/scratch/jcu/cv/lightfield/HCI/old/monasRoom/lf.h5'
    scene_path = '/scratch/jcu/cv/lightfield/HCI/others/statue.h5'
    scene_path = '/scratch/jcu/cv/lightfield/HCI/full_data/additional/town'
    scene_path = '/scratch/jcu/cv/lightfield/Inria/Inria_syn_lf_datasets/DLFD/Bowl_chair_dense'

    LF, disparity = read_newhci(scene_path)
    pdb.set_trace()
