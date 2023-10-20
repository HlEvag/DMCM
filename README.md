## Cross-Domain Meta-Learning under Dual Adjustment Mode for Few-Shot Hyperspectral Image Classification, TGRS, 2023.
This is a code demo for the paper: [Cross-Domain Meta-Learning under Dual Adjustment Mode for Few-Shot Hyperspectral Image Classification.](https://doi.org/10.1109/TGRS.2023.3320657)

<div align=center><img src="/Flowchart/Flowchart.tif" width="80%" height="80%"></div>
Fig. 1: Overall framework of the DMCM. The dual-adjustment mode-based cross-domain meta-learning framework comprises the following four parts: (a) feature extraction, (b) cross-domain FSL, (c) dual adjustment mode, and (d) target dataset classification.

## References
If you find this code helpful, please kindly cite:
```
@article{hu2023cross,
  title={Cross-Domain Meta-Learning under Dual Adjustment Mode for Few-Shot Hyperspectral Image Classification},
  author={Hu, Lei and He, Wei and Zhang, Liangpei and Zhang, Hongyan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```

Some of our references the projects, and we are very grateful for their research:
* [Graph Information Aggregation Cross-Domain Few-Shot Learning for Hyperspectral Image Classification, TNNLS, 2022.](https://github.com/YuxiangZhang-BIT/IEEE_TNNLS_Gia-CFSL)
* [Deep Cross-domain Few-shot Learning for Hyperspectral Image Classification, TGRS, 2022.](https://github.com/Li-ZK/DCFSL-2021)
* [Few-shot Learning with Class-Covariance Metric for Hyperspectral Image Classification, TIP, 2022.](https://github.com/B-Xi/TIP_2022_CMFSL)

## Requirements
CUDA Version = 11.7

Python = 3.9.7 

Pytorch = 1.12.0 

Sklearn = 0.24.2

Numpy = 1.20.0

Matplotlib = 3.4.3

Spectral = 0.22.4

## Dataset
1. Target domain data set: Indian Pines (IP)/ Pavia University (UP)/Pavia Center (PC)

You can download the hyperspectral datasets in mat format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes, and move the files to `./Datasets` folder.

2. Source domain data set: Chikusei

The source domain hyperspectral datasets (Chikusei) in mat format is available in: https://github.com/Li-ZK/DCFSL-2021. 
 
You can also download our preprocessed source domain data set (Chikusei_imdb_128.pickle) directly in pickle format, please move to the `./Datasets` folder. However, the data is too large to upload, if necessary, please contact my email: hulei.eva@whu.edu.cn.

An example dataset folder has the following structure:
```
Datasets
├── Chikusei_imdb_128.pickle
├── IP
│   ├── indian_pines.mat
│   ├── indian_pines_gt.mat
├── paviaU
│   ├── paviaU.mat
│   └── paviaU_gt.mat
└── paviaU
    ├── PaviaCenter.mat
    └── PaviaCenter_gt.mat
```

## Usage:
An example of DMCM method:
1. Download the required data set and move to folder`./Datasets`.
2. If you down the source domain data set (Chikusei) in mat format, you need to run the script `./Datasets/Generate_source_dataset.py` to generate preprocessed source domain data. 
3. Taking 5 labeled samples per class as an example, run `DMCM_IP/UP/PC.py`. 
 * `--test_lsample_num_per_class` denotes the number of labeled samples per class for the target domain data set.
 * `--tar_input_dim` denotes the number of bands for the target domain data set.
