# **[PointNetVLAD](https://arxiv.org/abs/1804.03492)**: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition [[Original Code]](https://github.com/cattaneod/PointNetVlad-Pytorch) (CVPR 2018)

## Evaluation
Please change the `EVAL_DATABASE_FILE`, `EVAL_QUERY_FILE` and `MODEL_FILENAME` in evaluate.py.
```
python evaluate.py
```

## Citation
If you find this project useful in your research, please cite these works:

```latex
@article{jung2024hetero,
    author = {Minwoo Jung and Wooseong Yang and Dongjae Lee and Hyeonjae Gil and Giseop Kim and Ayoung Kim},
    title ={HeLiPR: Heterogeneous LiDAR dataset for inter-LiDAR place recognition under spatiotemporal variations},
    journal = {The International Journal of Robotics Research},
    year = {2024}
}
```

```latex
@InProceedings{Uy_2018_CVPR,
    author = {Uy, Mikaela Angelina and Lee, Gim Hee},
    title = {PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
} 
```