# HeLiPR-Place-Recognition

## Introduction
This repository is created for comparing different place recognition methods on the HeLiPR dataset. We provide the code for the following methods:

| Methods| Complete |
|------|-----|
| PointNetVLAD |   To do   |
| MinkLoc3Dv2  |   To do   |
| LoGG3D-Net   |   To do   |
| CROSSLOC3D   | 24.09.24  |
| CASSPR       |   To do   |
| SOLID        |   To do   |
| HeLiOS       |   To do   |

- [**PointNetVLAD**](https://arxiv.org/abs/1804.03492): Deep Point Cloud Based Retrieval for Large-Scale Place Recognition [[Orignal Code]](https://github.com/cattaneod/PointNetVlad-Pytorch) (2018 CVPR)
- [**MinkLoc3Dv2**](https://arxiv.org/pdf/2203.00972v1): Improving Point Cloud Based Place Recognition with Ranking-based Loss and Large Batch Training [[Orignal Code]](https://github.com/jac99/MinkLoc3Dv2?tab=readme-ov-file) (2022 ICPR)
- [**LoGG3D-Net**](https://arxiv.org/abs/2109.08336): Locally Guided Global Descriptor Learning for 3D Place Recognition [[Orignal Code]](https://github.com/csiro-robotics/LoGG3D-Net) (2021 ICRA)
- [**CrossLoc3D**](https://arxiv.org/abs/2303.17778): Aerial-Ground Cross-Source 3D Place Recognition [[Orignal Code]](https://github.com/rayguan97/crossloc3d) (2023 ICCV)
- [**CASSPR**](https://arxiv.org/abs/2211.12542): Cross Attention Single Scan Place Recognition [[Orignal Code]](https://github.com/Yan-Xia/CASSPR) (2023 ICCV)
- [**SOLID**](https://arxiv.org/abs/2408.07330): Spatially Organized and Lightweight Global Descriptor for FOV-constrained LiDAR Place Recognition [[Orignal Code]](https://github.com/sparolab/solid) (2024 RA-L)
- **HeLiOS**: Heterogeneous LiDAR Place Recognition via Overlap-based Learning and Local Spherical Transformer [[Orignal Code]](https://github.com/minwoo0611/HeLiOS) (2025 ICRA submission)


## Comparison in HeLiPR dataset
<img src="assets/Table.png" width=full>

This table represents the comparison of different methods on the HeLiPR dataset. The evaluation is done on the 3D point cloud data of the HeLiPR dataset. The number of points in each scan is 8192.
We utilize the identical parameter settings for all the methods to ensure a fair comparison. 
Average Recall@1 and Average Recall@5 are used as evaluation metrics, and each number indicates the average of the results from the 4~6 sequences in the HeLiPR dataset.

For example, Ouster - Narrow indicates the results from narrow field-of-view (FOV) LiDAR data from the Ouster sensor.
- DB: Sequence01-Ouster
- Query: Sequence01-Aeva, Sequence01-Livox, Sequence02-Aeva, Sequence02-Livox, Sequence03-Aeva, Sequence03-Livox

On the other hand, Aeva - Wide indicates the results from wide FOV LiDAR data from the Ouster sensor.
- DB: Sequence01-Aeva
- Query: Sequence01-Ouster, Sequence01-Velodyne, Sequence02-Ouster, Sequence02-Velodyne, Sequence03-Ouster, Sequence03-Velodyne

Especially, for the Bridge sequence, we grouped Bridge01-04 and Bridge02-03 to obtain the sufficient overlap between the DB and query scans.
Therefore, for the Bridge01-04 cases, Ouster-Narrow indicates as follows:
- DB: Sequence01-Ouster
- Query: Sequence01-Ouster, Sequence01-Velodyne, Sequence02-Ouster, Sequence02-Velodyne

## Usage
(2024/09/24) Now, we only test the validation code for each method. The training code will be tested and updated soon.

Please download the validation dataset from [here](https://drive.google.com/drive/folders/10wXhjOnKlhkxm3a1Td34YdtNJCRFxIoZ?usp=drive_link). This link contains the sampled point cloud data from the HeLiPR dataset (Roundabout, Town and Bridge). The number of points in each scan is 8192 and all scans are sampled with 5m interval.
Furthermore, the chkeckpoint files for each method and overlap matrix files are also included in the link.
If you want to test the custom setting, please utilize [HeLiPR-Pointcloud-Toolbox](https://github.com/minwoo0611/HeLiPR-Pointcloud-Toolbox).

Each sequence, overlap and checkpoint files should be located in the data_validation/, data_overlap/ and data_ckpt/ folders, respectively.

Finally, the file structure should be as follows:
```
HeLiPR-Place-Recognition
├── model_X
├── data_validation
│   ├── SequenceA-Sensor1
│       ├── LiDAR
│           ├── 000000.bin
│           ├── 000001.bin
│           ├── ...
│       ├── trajectory.csv
│   ├── SequenceB-Sensor2
│       ├── LiDAR
│           ├── 000000.bin
│           ├── 000001.bin
│           ├── ...
│       ├── trajectory.csv
│              ...
├── data_overlap
│   ├── overlap_matrix_validation_SequenceA.txt
│   ├── overlap_matrix_validation_SequenceB.txt
│              ...
├── data_ckpt
│   ├── X_ckpt.pth
│   ├── Y_ckpt.pth
│              ...
```
Then, you can run the validation code for each method as follows:

We provide the Dockerfile for each method. You can build the docker image and run the code in the docker container.
1. Build the docker image and run the container
```bash
docker build -t helipr_evaluation .
docker run --gpus all -dit --env="DISPLAY" --net=host --ipc=host --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /:/mydata --volume /dev/:/dev/ helipr_evaluation:latest /bin/bash
```
You don't need to run the docker container with the above command. You can run the container with your own setting.

2. Clone the repository
```bash
git clone https://github.com/minwoo0611/HeLiPR-Place-Recognition
cd HeLiPR-Place-Recognition
```
3. Generate the pickle files for the HeLiPR dataset
```bash
python generate_test_sets.py
```
before running the code, please download the validation dataset from the link above and put the data in each folder.
Based on the location, please change these variables
- base_path
- overlap_matrix (file name)
- location (sequence name)
- db_folder (which sequence-sensor is used for the database)
- query_folder (which sequence-sensor is used for the query) 

Then, you can see the two files in base_path: helipr_validation_db.pickle and helipr_validation_query.pickle.

4. Run the validation code for each method
```bash
cd model_X
(Please check the README.md in each folder)
```

## Citation
If you find this repository useful, please cite the following papers:
```
@article{jung2024heteropr,
  author={Minwoo Jung and Sangwoo Jung and Hyeonjae Gil and Ayoung Kim}
  title={HeLiOS: Heterogeneous LiDAR Place Recognition via Overlap-based Learning and Local Spherical Transformer}}
```

```
@article{jung2024hetero,
author = {Minwoo Jung and Wooseong Yang and Dongjae Lee and Hyeonjae Gil and Giseop Kim and Ayoung Kim},
title ={HeLiPR: Heterogeneous LiDAR dataset for inter-LiDAR place recognition under spatiotemporal variations},
journal = {The International Journal of Robotics Research},
year = {2024}}
```

## Contact
If you have any questions, please contact here (moonshot@snu.ac.kr) or make an issue in this repository.
