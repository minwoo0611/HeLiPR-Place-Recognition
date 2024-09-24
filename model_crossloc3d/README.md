# [**CrossLoc3D**](https://arxiv.org/abs/2303.17778): Aerial-Ground Cross-Source 3D Place Recognition [[Orignal Code]](https://github.com/rayguan97/crossloc3d) (2023 ICCV)

## Evaluation
Please change the path in configs/dataset_cfg/helipr_cfg.py to your folder path.
```
CUDA_VISIBLE_DEVICES=0 python main.py ./configs/helipr.py --mode val --resume_from ../data_ckpt/crossloc_ckpt.pth
```

## Citation
If you find this project useful in your research, please cite these works:

```latex
@article{jung2024hetero,
author = {Minwoo Jung and Wooseong Yang and Dongjae Lee and Hyeonjae Gil and Giseop Kim and Ayoung Kim},
title ={HeLiPR: Heterogeneous LiDAR dataset for inter-LiDAR place recognition under spatiotemporal variations},
journal = {The International Journal of Robotics Research},
year = {2024}}
```

```latex
@InProceedings{Guan_2023_ICCV,
    author    = {Guan, Tianrui and Muthuselvam, Aswath and Hoover, Montana and Wang, Xijun and Liang, Jing and Sathyamoorthy, Adarsh Jagan and Conover, Damon and Manocha, Dinesh},
    title     = {CrossLoc3D: Aerial-Ground Cross-Source 3D Place Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
}

```