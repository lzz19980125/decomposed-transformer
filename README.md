# Decomposed Transformer with Frequency Attention for Multivariate Time Series Anomaly Detection

**Unofficial** Python implementation of the decomposed transformer algorithm:  
Qin, Shuxin, et al. Decomposed transformer with frequency attention for multivariate time series anomaly detection. *2022 IEEE International Conference on Big Data (Big Data)*. IEEE, 2022. 

## Get Started

1. Install Python 3.6, PyTorch >= 1.4.0. 

2. Download data. You can obtain four benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1KOQvV2nR6Y9tIkB4XELHA1CaV1F8LrZ6). **All the datasets are well pre-processed**. 

```
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
bash ./scripts/SWAT.sh
```

## Citation

```bibtex
@inproceedings{qin2022decomposed,
  title={Decomposed transformer with frequency attention for multivariate time series anomaly detection},
  author={Qin, Shuxin and Zhu, Jing and Wang, Dan and Ou, Liang and Gui, Hongxin and Tao, Gaofeng},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={1090--1098},
  year={2022},
  organization={IEEE}
}
```

