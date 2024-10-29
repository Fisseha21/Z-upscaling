"""
Created on Tue May  14 09:53:15 2024

@author: Fisseha Ferede
       : fissehaad@gmail.com"""
import os
from glob import glob
import os.path as osp
import tifffile
import cv2
import csv
from skimage.metrics import structural_similarity as ssim
import numpy as np
import imageio
from scipy.ndimage import zoom
from natsort import natsorted
import tensorflow as tf
from absl import logging

class Bicubic:
    def __init__(self, input_path: str, output_path: str, channel: str, save_vol: bool, vol_path: str, upsample_factor:int):
        self.input_path = input_path
        self.output_path = output_path
        self.channel = channel
        self.vol_save = save_vol
        self.vol_path = vol_path
        self.upsample_factor  = upsample_factor
    def read_png(self, image_path):
        image = tf.io.read_file(image_path)  # Assuming RGB images
        image = tf.io.decode_png(image)
        if self.channel == 'gray':
            image = tf.reduce_mean(image, axis=-1)
        return image

    def read_full_volume(self, file_path):
        images_list = natsorted(glob(osp.join(file_path, '*.png')))
        image_tensors = [self.read_png(path) for path in images_list]

        expanded_tensors = [tf.expand_dims(tensor, axis=0) for tensor in image_tensors]
        concatenated_image = tf.concat(expanded_tensors, axis=0)

        return concatenated_image
    def bicubic_interpolation(self, volume, number_of_frames):
        return tf.image.resize(volume, [number_of_frames, tf.shape(volume)[2]], method=tf.image.ResizeMethod.BICUBIC)

    def __call__(self, output_frame):
        concatenated_images = self.read_full_volume(self.input_path)

        if self.channel != 'gray':
            concatenated_images = tf.transpose(concatenated_images, [1, 0, 2, 3])
            resized_data = (tf.transpose(self.bicubic_interpolation(concatenated_images, output_frame), [1, 0, 2, 3])).numpy()
        else:
            resized_data = self.bicubic_interpolation(concatenated_images, output_frame).numpy()

        #print('bicubic_resized_size', resized_data.shape)
        os.makedirs(self.output_path, exist_ok=True)
        if self.vol_save:
            os.makedirs(self.vol_path, exist_ok=True)
            tifffile.imwrite(self.vol_path + '/FILM-BC_interpolated_x' + str(self.upsample_factor)+'.tiff', resized_data, bigtiff=True)
        logging.info('bicubic_resized_size %s', resized_data.shape)

        for i in range(resized_data.shape[0]):

            name = self.output_path + '/' + 'frame' + str(i).zfill(3) + '.png'
            cv2.imwrite(name, resized_data[i])



