# **[LoGG3D-Net](https://arxiv.org/abs/2109.08336)**: Locally Guided Global Descriptor Learning for 3D Place Recognition [[Original Code]](https://github.com/csiro-robotics/LoGG3D-Net) (ICRA 2021)

## Evaluation
Please change the `--helipr_dir`, `--eval_task`, `--overlap_path`, `--eval_seq_q` and `--eval_seq_db` in config/eval_config.py.
```
python3 evaluation/evaluate.py
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
@inproceedings{vid2022logg3d,
    title={LoGG3D-Net: Locally Guided Global Descriptor Learning for 3D Place Recognition},
    author={Vidanapathirana, Kavisha and Ramezani, Milad and Moghadam, Peyman and Sridharan, Sridha and Fookes, Clinton},
    booktitle={2022 International Conference on Robotics and Automation (ICRA)},
    pages={2215--2221},
    year={2022}
}


```