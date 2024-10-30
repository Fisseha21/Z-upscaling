# Z-upscaling
This repository contains the source code for our paper:

[Z-upscaling: Optical Flow Guided Frame Interpolation for Isotropic Reconstruction of 3D EM Volumes](https://arxiv.org/pdf/2410.07043)<br/>
Fisseha A. Ferede, Ali Khalighifar, Jaison John, Krishnan Venkataraman, Khaled Khairy<br/>

<img src="Z-upscaling.png">

## Demo

<style type="text/css">
    .image-container {
        display: flex; /* Use flexbox for layout */
        justify-content: center; /* Center items horizontally */
        gap: 20px; /* Space between items */
        flex-wrap: wrap; /* Allow items to wrap */
        min-width: 700px; /* Minimum width for the container */
    }
    .captioned-image {
        text-align: center; /* Center text below images */
        width: 220px; /* Set a fixed width for image blocks */
    }
    img {
        max-width: 100%; /* Make images responsive */
        height: auto; /* Maintain aspect ratio */
    }
</style>

<section class="products">
    <p class="image-container">
        <div class="captioned-image">
            <img src="https://github.com/Fisseha21/Z-upscaling/blob/main/demos/Input_downsampledFIB25_vol.gif" alt="Input Downsampled Volume">
            <div style="margin-top: 8px;">Input Downsampled Volume</div>
        </div>

        <div class="captioned-image">
            <img src="https://github.com/Fisseha21/Z-upscaling/blob/main/demos/zUpx8_FIB25_vol.gif" alt="Upscaled Volume">
            <div style="margin-top: 8px;">Upscaled Volume (Z × 8)</div>
        </div>

        <div class="captioned-image">
            <img src="https://github.com/Fisseha21/Z-upscaling/blob/main/demos/GT_FIB25_vol.gif" alt="Ground Truth Volume">
            <div style="margin-top: 8px;">Ground Truth Volume</div>
        </div>
    </p>
</section>







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