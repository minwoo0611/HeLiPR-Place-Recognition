# **[MinkLoc3Dv2](https://arxiv.org/pdf/2203.00972v1)**: Improving Point Cloud Based Place Recognition with Ranking-based Loss and Large Batch Training [[Original Code]](https://github.com/jac99/MinkLoc3Dv2) (ICPR 2022)

## Evaluation
Please change the `eval_database_files`, `eval_query_files` and `DATASET_FOLDER` in evaluate.py.
```
export PYTHONPATH=$PYTHONPATH:/PATH/HeLiPR-Place-Recognition/model_minkloc3dv2
cd eval
python3 eval/evaluate.py --config config/config_helipr.txt --model_config models/minkloc3dv2.txt --weights ../data_ckpt/minkloc3dv2_ckpt.pth
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
@inproceedings{komorowski2022improving,
    title={Improving point cloud based place recognition with ranking-based loss and large batch training},
    author={Komorowski, Jacek},
    booktitle={2022 26th international conference on pattern recognition (ICPR)},
    pages={3699--3705},
    year={2022},
    organization={IEEE}
}

```