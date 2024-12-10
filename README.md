# traffic-pipeline

[**Paper**](https://shaadalam9.github.io/publications/traffic-pipeline)

## Overview
Recent advances in GAN-based architectures haveled to innovative methods for image transformation. The lack ofdiverse environmental data, such as different lighting conditionsand seasons in public data, prevents researchers from effectivelystudying the difference in driver and road user behaviourunder varying conditions. This study introduces a deep learningpipeline that combines CycleGAN-turbo and Real-ESRGAN toimprove video transformations of the traffic scene. Evaluatedusing dashcam videos from London, Hong Kong, and LosAngeles, our pipeline shows a 7.97% improvement in T-SIMMfor temporal consistency compared to CycleGAN-turbo for night-to-day transformation for Hong Kong. PSNR and VPQ scoresare comparable, but the pipeline performs better in DINOstructure similarity and KL divergence, with up to 153.49%better structural fidelity in Hong Kong compared to Pix2Pixand 107.32% better compared to ToDayGAN. This approachdemonstrates better realism and temporal coherence in day-to-night, night-to-day, and clear-to-rainy transitions.


## Usage of the code
The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code 😍😄 For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

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

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Data
### Dashcam Videos Used in the Study

This project utilizes dashcam videos from various locations. The following table lists the video links along with the specific timestamps from which the footage was extracted for the study:

| **Location**        | **Video Link**                                                                 | **Timestamps**       |
|---------------------|------------------------------------------------------------------------------|----------------------|
| **London (UK)**     | [Day](https://www.youtube.com/watch?v=QI4_dGvZ5yE)                      | 5:00 till 5:20       |
|                     | [Night](https://www.youtube.com/watch?v=mEXVBiT1eAM)                      | 22:20 till 22:40     |
| **Hong Kong**       | [Day](https://www.youtube.com/watch?v=ULcuZ3Q02SI)                      | 6:20 till 6:40       |
|                     | [Night](https://www.youtube.com/watch?v=XaR6qEt-BIY)                      | 25:40 till 26:00     |
| **Los Angeles (CA, USA)** | [Day](https://www.youtube.com/watch?v=4uhMg5na888)                 | 16:10 till 16:30     |
|                     | [Night](https://www.youtube.com/watch?v=eR5vsN1Lq4E)                      | 39:00 till 39:20     |

### Note:
- The timestamps indicate the portion of the video used in the study.

## Results

### Pipeline Architecture

<div>
<p align="center">
<img src='figures/architecture.png' align="center" width=1000px>
</p>
</div>


### Comparison with CycleGAN-turbo

<div>
<p align="center">
<img src='figures/compare.png' align="center" width=1000px>
</p>
</div>

#### CycleGAN-turbo Trained Model
- **Paper**: [CycleGAN-turbo: A Faster Approach to Image-to-Image Translation](https://arxiv.org/abs/2403.12036)
- **Code Repository**: [GitHub - img2img-turbo](https://github.com/GaParmar/img2img-turbo)


### Comparison for Day-to-Night Translation with Other Trained Models

<div>
<p align="center">
<img src='figures/compare_merged.png' align="center" width=1000px>
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


