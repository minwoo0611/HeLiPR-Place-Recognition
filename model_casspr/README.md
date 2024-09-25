# **[CASSPR](https://arxiv.org/abs/2211.12542)**: Cross Attention Single Scan Place Recognition [[Original Code]](https://github.com/Yan-Xia/CASSPR) (ICCV 2023)

## Evaluation
Please change the `dataset_folder`, `eval_database_files` and `eval_query_files` in configs/config_helipr.txt.
```
cd eval/
CUDA_VISIBLE_DEVICES=0  python evaluate.py --config ../config/config_helipr.txt --model_config ../config/model_config_helipr.txt --weights ../../data_ckpt/casspr_ckpt.pth
```
We observed a decline in model performance when using a large number of layers. To address this, we opted for a more efficient architecture with 3 self-attention layers, similar to what was used in the USYD dataset. Additionally, since each heterogeneous LiDAR sensor has its own unique characteristics, we found that training the model for long epochs led to instability. As a result, we limited the training to 35 epochs. Beyond 35-40 epochs, the distance between descriptors collapsed to zero. We also adhered to the default parameters from the original paper in most cases.

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
@InProceedings{Xia_2023_ICCV,
    author = {Xia, Yan and Gladkova, Mariia and Wang, Rui and Li, Qianyun and Stilla, Uwe and Henriques, Joao F and Cremers, Daniel},
    title = {CASSPR: Cross Attention Single Scan Place Recognition },
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2023},
    organization={IEEE}
}
```