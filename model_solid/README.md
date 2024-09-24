# **[SOLID](https://arxiv.org/abs/2408.07330)**: Spatially Organized and Lightweight Global Descriptor for FOV-constrained LiDAR Place Recognition [[Original Code]](https://github.com/sparolab/solid) (RA-L 2024)

## Evaluation
Please update the `dataset_folder` path in `evaluate_solid.py` to point to your data directory.

```bash
python evaluate_solid.py
```

**Note:** Unlike other methods, **SOLiD** does not require a checkpoint file for evaluation. Instead, it utilizes the following parameters, set to the maximum values across all sensors to generate consistent descriptors:

- `fov_u = 38.4` (upper field of view limit from Livox Avia)
- `fov_d = -38.4` (lower field of view limit from Livox Avia)
- `num_height = 128` (from OS2-128 sensor)
- `max_length = 100` (based on experimental settings)

These parameters ensure that the descriptors are consistent across different sensors by accommodating the widest possible field of view and resolution.

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
  @article{kim2024narrowing,
    title={Narrowing your FOV with SOLiD: Spatially Organized and Lightweight Global Descriptor for FOV-constrained LiDAR Place Recognition},
    author={Kim, Hogyun and Choi, Jiwon and Sim, Taehu and Kim, Giseop and Cho, Younggun},
    journal={IEEE Robotics and Automation Letters},
    year={2024},
    publisher={IEEE}
  }

```