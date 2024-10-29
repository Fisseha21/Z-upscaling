"""
Created on Tue May  14 09:53:15 2024

@author: Fisseha Ferede
       : fissehaad@gmail.com"""

import imageio
from glob import glob
import os.path as osp
import numpy as np
from natsort import natsorted
import tifffile
import cv2
import os
import cv2
from scipy.ndimage import zoom
import tensorflow as tf
from absl import logging

class VolSlicer:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
    def get_tiff_bit_depth(self, tiff_file):
        with tifffile.TiffFile(tiff_file) as tif:
            for page in tif.pages:
                bit_depth = page.bitspersample
                if bit_depth == 8:
                    return '8-bit'
                elif bit_depth == 16:
                    return '16-bit'
                else:
                    return f"Unsupported bit depth: {bit_depth}-bit"
    def read_tiff_files_from_directory(self, directory):
        tiff_file_list = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
                tiff_path = os.path.join(directory, filename)
                tiff_file_list.append(tiff_path)
        return tiff_file_list

    def __call__(self):
        tiff_path = self.read_tiff_files_from_directory(self.input_path)

        logging.info('length %i', len(tiff_path))

        path_to_2d_slices = []

        for i in range(len(tiff_path)):  #traverse through all the tiff volumes listed, slice the into 2D png images and save them in unique file
            vol = tifffile.imread(tiff_path[i])

            basename, _ = os.path.splitext(os.path.basename(tiff_path[i]))
            path = f'{self.output_path}/{basename}'

            if not tf.io.gfile.exists(path):
                path_to_2d_slices.append(path)

                for j in range(vol.shape[0]):
                    #logging.info('Path didnt exist!!! %i', vol.shape[0])
                    img = vol[j]
                    os.makedirs(path, exist_ok=True)

                    name = path + '/' + 'slice' + str(j).zfill(3) + '.png'

                    bit_type = self.get_tiff_bit_depth(tiff_path[i])

                    if bit_type != '8-bit':
                        image_16bit = img
                        img = ((image_16bit - np.min(image_16bit)) / (np.max(image_16bit) - np.min(image_16bit)) * 255).astype(
                        np.uint8)

                    cv2.imwrite(name, np.uint8(img))
        logging.info('Slicer is DONE!!!')
        return path_to_2d_slices