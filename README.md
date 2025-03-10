# [TPAMI 2024] Learning to Holistically Detect Bridges from Large-Size VHR Remote Sensing Imagery

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://luo-z13.github.io/GLH-Bridge-page/) 



<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="GLH-Bridge Overview">
</p>

<!-- ## Authors -->

 [Yansheng Li](https://scholar.google.com.hk/citations?user=wn9hc6UAAAAJ&hl=zh-CN&oi=ao) | 
[Junwei Luo*](https://github.com/Luo-Z13) | 
[Yongjun Zhang*](http://jszy.whu.edu.cn/zhangyongjun/en/index.htm) | 
[Yihua Tan](http://faculty.hust.edu.cn/tanyihua/zh_CN/zdylm/2416706/list/index.htm) | 
[Jin-Gang Yu](https://www2.scut.edu.cn/bci/2018/1005/c18566a287654/page.htm) | 
[Song Bai](https://songbai.site/)  
\* Indicates corresponding authors

---

## Table of Contents
- [Resource](#resource)
- [Introduction](#introduction)
- [GLH-Bridge Dataset](#glh-bridge-dataset)
- [Code](#code)
- [Citation](#citation)

---

## Resource

- **Dataset@Baidu Drive**: [Link](https://pan.baidu.com/s/1tXpFukMkOGHXPsSKPRRf3w?pwd=4zgx)  
- **Dataset@Hugging Face**: [Link](https://huggingface.co/datasets/ll-13/GLH-Bridge)  
- **Evaluation**: [Codabench Evaluation](https://www.codabench.org/competitions/3371/) 
 - **Weight**: [HBD-Net](https://drive.google.com/file/d/1wZcx4-c1MQ75qrKhRcS1Som0sywWctuP/view?usp=drive_link) 

## Introduction

Bridges represent critical infrastructure components, serving as fundamental transportation facilities that traverse various landscapes. They hold substantial significance in the domains like civil transportation and disaster relief efforts. To ensure the visibility and integrity of bridges, it is essential to perform **holistic bridge detection in large-size very-high-resolution (VHR) remote sensing images (RSIs)**. 

However, large-scale datasets and methods suitable for holistic bridge detection in large-sisze RSIs have yet to be explored. This work aims to addresses the challenge of **holistic bridge detection** in large-size VHR RSIs. Key contributions include:
- **GLH-Bridge**: A large-scale dataset with 6,000 globally sampled VHR images (from 2,048 x 2,048 to 16,384 x 16,384 pixels) and 59,737 manually annotated bridges.
- **HBD-Net**: A novel method for holistic bridge detection.
- Comprehensive benchmarks for both Oriented Bounding Box (OBB) and Horizontal Bounding Box (HBB) tasks.




---

## GLH-Bridge Dataset
### Key Features
🌍 **Diverse Coverage**  
6,000 images spanning global geographic locations with varied backgrounds (vegetation, riverbeds, roads).

🖼️ **Very-High-Resolution**  
Image sizes range from 2,048×2,048 to 16,384×16,384 pixels.

📊 **Rich Annotations**  
59,737 bridges annotated with both OBB and HBB labels, featuring:
- Extreme aspect ratios
- Multi-scale instances
- Varied instance densities

## Code


Please employ the following scripts for training and inference:

- **Training**:
  - Multi-resolution data splitting: [`tools/split_code.py`](./tools/split_code.py)
  - Load checkpoint for stage-by-stage training: [`tools/loadckpt_backbone.py`](./tools/loadckpt_backbone.py)
  - Training script: [`train_dist_mmrot.sh`](./train_dist_mmrot.sh)

- **Inference**:
  - Test script: [`test_mmrot_all.sh`](./test_mmrot_all.sh)

### Citation

If you find this work helpful for your research, please consider citing our papers:

```bibtex
@article{li2024learning,
    title={Learning to Holistically Detect Bridges From Large-Size VHR Remote Sensing Imagery},
    author={Li, Yansheng and Luo, Junwei and Zhang, Yongjun and Tan, Yihua and Yu, Jin-Gang and Bai, Song},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    volume={44},
    number={11},
    pages={7778--7796},
    year={2024},
    publisher={IEEE}
}


@article{li2024scene,
    title={STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery},
    author={Li, Yansheng and Wang, Linlin and Wang, Tingzhu and Yang, Xue and Luo, Junwei and Wang, Qi and Deng, Youming and Wang, Wenbin and Sun, Xian and Li, Haifeng and Dang, Bo and Zhang, Yongjun and Yu, Yi and Yan Junchi},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2024},
    publisher={IEEE}}
}
