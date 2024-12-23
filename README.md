# traffic-pipeline

## Overview
Recent advances in GAN-based architectures haveled to innovative methods for image transformation. The lack of diverse environmental data, such as different lighting conditionsand seasons in public data, prevents researchers from effectively studying the difference in driver and road user behaviourunder varying conditions. This study introduces a deep learning pipeline that combines CycleGAN-turbo and Real-ESRGAN to improve video transformations of the traffic scene. Evaluated using dashcam videos from London, Hong Kong, and LosAngeles, our pipeline shows a 7.97% improvement in T-SIMM for temporal consistency compared to CycleGAN-turbo for night-to-day transformation for Hong Kong. PSNR and VPQ scoresare comparable, but the pipeline performs better in DINO structure similarity and KL divergence, with up to 153.49% better structural fidelity in Hong Kong compared to Pix2Pix and 107.32% better compared to ToDayGAN. This approach demonstrates better realism and temporal coherence in day-to-night, night-to-day, and clear-to-rainy transitions.


## Usage of the code
The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code 😍😄 For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Citation
If you use the coupled sim for academic work please cite the following paper:

> Alam, M.S., Parmar, S.H., Martens, M.H., & Bazilinskyy, P. (2025). Deep Learning Approach for Realistic Traffic Video Changes Across Lighting and Weather Conditions. 7th International Conference on Information and Computer Technologies (ICICT). Hilo, Hawaii, USA. 

## Getting Started
Tested with Python 3.9.19. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows:

**Step 1:**

Clone the repository
```command line
git clone https://github.com/Shaadalam9/traffic-pipeline
```

**Step 2:**

Create a new virtual environment
```command line
python -m venv venv
```

**Step 3:**

Activate the virtual environment
```command line
source venv/bin/activate
```

On Windows use
```command line
venv\Scripts\activate
```

**Step 4:**

Install dependencies
```command line
pip install -r requirements.txt
```

**Step 5:**

Download the supplementary material from [4TU Research Data](https://doi.org/10.4121/80c664cb-a4b5-4eb1-bc1c-666349b1b927) and save them in the current folder.

**Step 6:**

Run the main.py script
```command line
python3 main.py
```

### Configuration of project
Configuration of the project needs to be defined in `traffic-pipeline/config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
- **`data`**: Specifies the location of the video files.
- **`transformation`**: Defines the type of transformation to apply.  
  - **Options**:  
    1. `day_to_night`: Convert daytime images to nighttime.  
    2. `night_to_day`: Convert nighttime images to daytime.  
    3. `style_transfer`: Apply a style transfer transformation.  
- **`plotly_template`**: Template used to style graphs in the analysis (e.g., `plotly_white`, `plotly_dark`).

## Data
### Dashcam Videos Used in the Study

This project utilizes dashcam videos from various locations. The following table lists the video links along with the specific timestamps from which the footage was extracted for the study:

| **Location**             | **Day**                                                                                       | **Night**                                                                                     | **Timestamps**         |
|--------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|------------------------|
| **London (UK)**          | [![Day](ReadmeFiles/thumbnail_london_day.png)](https://www.youtube.com/watch?v=QI4_dGvZ5yE)   | [![Night](ReadmeFiles/thumbnail_london_night.png)](https://www.youtube.com/watch?v=mEXVBiT1eAM) | 5:00 - 5:20 (Day) <br> 22:20 - 22:40 (Night) |
| **Hong Kong**            | [![Day](ReadmeFiles/thumbnail_hk_day.png)](https://www.youtube.com/watch?v=ULcuZ3Q02SI)       | [![Night](ReadmeFiles/thumbnail_hk_night.png)](https://www.youtube.com/watch?v=XaR6qEt-BIY)  | 6:20 - 6:40 (Day) <br> 25:40 - 26:00 (Night) |
| **Los Angeles (CA, USA)**| [![Day](ReadmeFiles/thumbnail_la_day.png)](https://www.youtube.com/watch?v=4uhMg5na888)       | [![Night](ReadmeFiles/thumbnail_la_night.png)](https://www.youtube.com/watch?v=eR5vsN1Lq4E)  | 16:10 - 16:30 (Day) <br> 39:00 - 39:20 (Night) |



### Note:
- The timestamps indicate the portion of the video used in the study.

## Results

### Pipeline Architecture

<div>
<p align="center">
<img src='ReadmeFiles/architecture.png' align="center" width=1000px>
</p>
</div>


### Comparison with CycleGAN-turbo

<div>
<p align="center">
<img src='ReadmeFiles/compare.png' align="center" width=1000px>
</p>
</div>

#### CycleGAN-turbo Trained Model
- **Paper**: [CycleGAN-turbo: A Faster Approach to Image-to-Image Translation](https://arxiv.org/abs/2403.12036)
- **Code Repository**: [GitHub - img2img-turbo](https://github.com/GaParmar/img2img-turbo)


### Comparison for Day-to-Night Translation with Other Trained Models

<div>
<p align="center">
<img src='ReadmeFiles/compare_merged.png' align="center" width=1000px>
</p>
</div>

---

## Trained Models Used in the Comparison Study

The following trained models were utilized for the comparison study. The respective papers and weight model links are provided below:

| **Model**    | **Paper**                                                                                   | **Weight Model**                                                                                 |
|--------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| **CycleGAN** | [Paper](https://arxiv.org/abs/1703.10593)                                                   | [Weight Model](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)                        |
| **HEDNGAN**  | [Paper](https://arxiv.org/abs/2309.16351)                                                   | [Weight Model](https://github.com/mohwald/gandtr)                                              |
| **Pix2Pix**  | [Paper](https://arxiv.org/abs/1611.07004)                                                   | [Weight Model](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)                        |
| **ToDayGAN** | [Paper](https://arxiv.org/pdf/1809.09767)                                                   | [Weight Model](https://github.com/AAnoosheh/ToDayGAN)                                          |

### Notes:
- Each model's paper outlines the theoretical framework and methodology behind its functionality.
- The weight model links direct you to the repositories where the pre-trained weights used in this study are available.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgement:
Code derived from [img2img-turbo](https://github.com/GaParmar/img2img-turbo) and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
