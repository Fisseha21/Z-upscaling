# Z-upscaling
This repository contains the source code for our paper:

[Z-upscaling: Optical Flow Guided Frame Interpolation for Isotropic Reconstruction of 3D EM Volumes](https://arxiv.org/pdf/2410.07043)<br/>
Fisseha A. Ferede, Ali Khalighifar, Jaison John, Krishnan Venkataraman, Khaled Khairy<br/>

<img src="Z-upscaling.png">

## Demo

<div style="display: flex; justify-content: center; gap: 20px;">
   <figure style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0;">
      <img src="https://github.com/Fisseha21/Z-upscaling/blob/main/demos/Input_downsampledFIB25_vol.gif" width="200" height="200" alt="Input Downsampled Volume">
      <figcaption>Input Downsampled Volume</figcaption>
   </figure>

   <figure style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0;">
      <img src="https://github.com/Fisseha21/Z-upscaling/blob/main/demos/zUpx8_FIB25_vol.gif" width="200" height="200" alt="Upscaled Volume">
      <figcaption>Upscaled Volume (Z × 8)</figcaption>
   </figure>

   <figure style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0;">
      <img src="https://github.com/Fisseha21/Z-upscaling/blob/main/demos/GT_FIB25_vol.gif" width="200" height="200" alt="Ground Truth Volume">
      <figcaption>Ground Truth Volume</figcaption>
   </figure>
</div>




## Evaluation

Download pretrained models [Saved models](https://drive.google.com/drive/folders/1vFvyuP4FdU8A0_Y0iA7CHSvlAFPH6StX?usp=sharing)
```Shell
python3 /Z-upscaling-main/eval/interpolator_cli.py \
   --pattern "/Z-upscaling-main/Demo/*" \
   --model_path /Z-upscaling-main/ModelPaths/test_run_ft_em_/saved_model_2M \
   --outputfile /Z-upscaling-main/Demo_out \
   --times_to_interpolate 8 \
   --output_volume "True" \
   --remove_sliced_volumes "False"

```