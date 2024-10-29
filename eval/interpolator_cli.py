# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified by Fisseha fferede (fissehaad[at]gmail.com)
r"""Runs the FILM frame interpolator on a pair of frames on beam.

This script is used evaluate the output quality of the FILM Tensorflow frame
interpolator. Optionally, it outputs a tiff volume of the interpolated frames.

A beam pipeline for invoking the frame interpolator on a set of directories
identified by a glob (--pattern). Each directory is expected to contain two
input frames that are the inputs to the frame interpolator. If a directory has
more than two frames, then each contiguous frame pair is treated as input to
generate in-between frames.

And/or (alternatively)

The interpolator takes .tiff volumes of depth N as inputs, slices these volumes as N 2D png images and saves them in
a directory with the same name as the original .tiff

E.g.  <root directory of the eval>/vol01.tiff
      <root directory of the eval>/vol02.tiff

      vol01.tiff and vol02.tiff with N and M number of z slices will be sliced and stored as:

      <root directory of the eval>/vol01/slice000.png
                                         slice001.png
                                              .
                                              .
                                         slice00N.png
      <root directory of the eval>/vol01/slice000.png
                                         slice001.png
                                              .
                                              .
                                         slice00M.png

The output video is stored to interpolator.mp4 in each directory. The number of
frames is determined by --times_to_interpolate, which controls the number of
times the frame interpolator is invoked. When the number of input frames is 2,
the number of output frames is 2^times_to_interpolate+1.

This expects a directory structure such as:
  <root directory of the eval>/01/frame1.png
                                  frame2.png
  <root directory of the eval>/02/frame1.png
                                  frame2.png
  <root directory of the eval>/03/frame1.png
                                  frame2.png

  and / or

  <root directory of the eval>/04.tiff
  <root directory of the eval>/05.tiff
  ...

And will produce:
  <root directory of the eval>/01/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  <root directory of the eval>/02/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  <root directory of the eval>/03/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  ...

And optionally will produce:
  <root directory of the eval>/01/interpolated.mp4
  <root directory of the eval>/02/interpolated.mp4
  <root directory of the eval>/03/interpolated.mp4
  ...

Usage example:
  python3 -m frame_interpolation.eval.interpolator_cli \
    --model_path <path to TF2 saved model> \
    --pattern "<root directory of the eval>/*" \
    --times_to_interpolate <Number of times to interpolate>
"""

import functools
import shutil
import os.path as osp
import os
from typing import List, Sequence
from glob import glob
import interpolator as interpolator_lib
import bicubic_upsampler
import tiff_slicer
#from . import util
import util
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import imageio
import tifffile
import cv2
# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='The pattern to determine the directories with the input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_OUTPUT_PATH = flags.DEFINE_string(
    name='outputfile',
    default=None,
    help='The path where newly interpolated frames are stored.')
_TIMES_TO_INTERPOLATE = flags.DEFINE_float(
    name='times_to_interpolate',
    default=5,
    help='The number of times to run recursive midpoint interpolation. '
    'The number of output frames will be 2^times_to_interpolate+1.')
_FPS = flags.DEFINE_integer(
    name='fps',
    default=30,
    help='Frames per second to play interpolated videos in slow motion.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=64,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='An int >= 1, number of patches along height, '
    'patch_height = height//block_height, should be evenly divisible.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='An int >= 1, number of patches along width, '
    'patch_width = width//block_width, should be evenly divisible.')    
_ERASE_SLICED_VOLUME = flags.DEFINE_string(
    name='remove_sliced_volumes',
    default=None,
    help='If false, deletes generated 2D slices of tiff volume inputs; otherwise, keeps the sliced tiff volumes in the input path')
# _OUTPUT_VOLUME = flags.DEFINE_boolean(
#     name='output_volume',
#     default=False,
#     help='If true, creates a tiff volume of the frames in the interpolated_frames/ '
#     'subdirectory')

_OUTPUT_VOLUME = flags.DEFINE_string(
    name='output_volume',
    default=None,
    help='If true, creates a tiff volume of the frames in the interpolated_frames/ '
    'subdirectory')

# Add other extensions, if not either.
_INPUT_EXT = ['png', 'jpg', 'jpeg']

def parse_string_to_boolean(input_str):
    normalized_str = input_str.strip().lower()
    if normalized_str == "true":
        return True
    elif normalized_str == "false":
        return False
    else:
        raise ValueError("Input string must be 'true' or 'false' (case insensitive).")

def _output_frames(frames: List[np.ndarray], frames_dir: str):
  """Writes PNG-images to a directory.

  If frames_dir doesn't exist, it is created. If frames_dir contains existing
  PNG-files, they are removed before saving the new ones.

  Args:
    frames: List of images to save.
    frames_dir: The output directory to save the images.

  """
  if tf.io.gfile.isdir(frames_dir):
    old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
    if old_frames:
      logging.info('Removing existing frames from %s.', frames_dir)
      for old_frame in old_frames:
        tf.io.gfile.remove(old_frame)
  else:
    tf.io.gfile.makedirs(frames_dir)
  for idx, frame in tqdm(
      enumerate(frames), total=len(frames), ncols=100, colour='green'):
    util.write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
  logging.info('Output frames saved in %s.', frames_dir)


class ProcessDirectory(beam.DoFn):
  """DoFn for running the interpolator on a single directory at the time."""

  def setup(self):
    self.interpolator = interpolator_lib.Interpolator(
        _MODEL_PATH.value, _ALIGN.value,
        [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
    #if _OUTPUT_VIDEO.value:
      #ffmpeg_path = util.get_ffmpeg_path()
      #media.set_ffmpeg(ffmpeg_path)

  def process(self, directory: str):
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
        for ext in _INPUT_EXT
    ]
    input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
    logging.info('Generating in-between frames for %s.', directory)

    up_x = np.floor(np.log(_TIMES_TO_INTERPOLATE.value)/np.log(2)).astype(int)
    down_x = np.ceil(np.log(_TIMES_TO_INTERPOLATE.value)/np.log(2)).astype(int)

    bicubic = True
    if _TIMES_TO_INTERPOLATE.value == 2**up_x: #Only FILM
        film_n = up_x
        bicubic =  False
    elif _TIMES_TO_INTERPOLATE.value - 2**up_x >= 2**down_x - _TIMES_TO_INTERPOLATE.value: #Upsample using Bicubic on FILM Results
        film_n = down_x
    else: #Downsample using Bicubic on FILM Results
        film_n = up_x

    basename, _ = os.path.splitext(os.path.basename(directory))

    frames = list(
        util.interpolate_recursively_from_files(
            input_frames, film_n, self.interpolator))
    _output_frames(frames, f'{_OUTPUT_PATH.value}/{basename}/film_interpolated_frames_x' + str(2**film_n))

    vol_path = f'{_OUTPUT_PATH.value}/{basename}/interpolated_vol'

    output_vol_flag = parse_string_to_boolean(_OUTPUT_VOLUME.value)

    if bicubic:
        logging.info('Bicubic Interpolation Running')
        l = len(frames)  # number of FILM interpolated frames
        m = (l - 1) / 2 ** film_n + 1  # original input size before FILM interpolation
        bc_n = (_TIMES_TO_INTERPOLATE.value - 1) * (m - 1) + m

        bicubic = bicubic_upsampler.Bicubic(f'{_OUTPUT_PATH.value}/{basename}/film_interpolated_frames_x' + str(2**film_n),
             f'{_OUTPUT_PATH.value}/{basename}/film-bic_interpolated_frames_x'+str(_TIMES_TO_INTERPOLATE.value), channel='gray',
                                            save_vol=output_vol_flag,
                                            vol_path=vol_path,
                                            upsample_factor = _TIMES_TO_INTERPOLATE.value)
        bicubic(bc_n)

    if output_vol_flag:
        #if bicubic:
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        logging.info('Length of interpolated frames %d', len(gray_frames))

        volume = np.stack(gray_frames, axis=0)
        logging.info('shape of volume array %s', volume.shape)

        os.makedirs(vol_path, exist_ok=True)
        tifffile.imwrite(vol_path + '/FILM_interpolated_x'+str(2**film_n)+'.tiff', volume, bigtiff=True)

def _run_pipeline() -> None:
    slicer = tiff_slicer.VolSlicer(os.path.dirname(f'{_PATTERN.value}'),
                                       os.path.dirname(f'{_PATTERN.value}'))
    path_to_2d_slices = slicer()

    logging.info('List of sliced directories %s', path_to_2d_slices)

    directories = tf.io.gfile.glob(_PATTERN.value)
    directories = [entry for entry in directories if tf.io.gfile.isdir(entry)]
    logging.info('List of all directories %s', directories)

    pipeline = beam.Pipeline('DirectRunner')
    (pipeline | 'Create directory names' >> beam.Create(directories)  # pylint: disable=expression-not-assigned
    | 'Process directories' >> beam.ParDo(ProcessDirectory()))

    result = pipeline.run()
    result.wait_until_finish()

    if parse_string_to_boolean(_ERASE_SLICED_VOLUME.value):
        for i in range(len(path_to_2d_slices)):
            shutil.rmtree(path_to_2d_slices[i])

    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    status_file_name = f'{_OUTPUT_PATH.value}/' + 'status_file.txt'
    #status_file_name = f'{_OUTPUT_PATH.value}/' + current_datetime + '.txt'

    with open(status_file_name, "w") as file:
        # Write some text to the file
        file.write("DONE")
        ##file.write("This is the second line.\n")

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _run_pipeline()


if __name__ == '__main__':
  app.run(main)
