import os
import sys
import glob
import random
import numpy as np
import logging
import json
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.o3d_tools import *
from utils.data_loaders.pointcloud_dataset import *

class HeliprDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from Helipr dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config.helipr_dir
        self.pnv_prep = config.pnv_preprocessing
        self.gp_rem = config.gp_rem
        self.int_norm = config.helipr_normalize_intensity

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing HeliprDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = config.helipr_data_split[phase]
        for drive_id in sequences:
            inames = self.get_all_scan_ids(drive_id)
            for query_id, start_time in enumerate(inames):
                self.files.append((drive_id, query_id)) # ex. ('KAIST/KAIST_01', 1000)

    def get_all_scan_ids(self, drive_id):
        sequence_path = self.root + drive_id + '/LiDAR/'
        fnames = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    def get_velodyne_fn(self, drive_id, query_id):
        sequence_path = self.root + drive_id + '/LiDAR/'
        fname = sorted(glob.glob(os.path.join(
            sequence_path, '*.bin')))[query_id]
        return fname

    def get_pointcloud_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, pc_id)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        xyzr[:, :3] = 100 * xyzr[:, :3]
        range = np.linalg.norm(xyzr[:, :3], axis=1)
        range_filter = np.logical_and(range > 0.1, range < 100)
        xyzr = xyzr[range_filter]
        if self.int_norm:
            xyzr[:, 3] = np.clip(xyzr[:, 3], 0, 1000) / 1000.0
        if self.gp_rem:
            not_ground_mask = np.ones(len(xyzr), bool)
            raw_pcd = make_open3d_point_cloud(xyzr[:, :3], color=None)
            _, inliers = raw_pcd.segment_plane(0.2, 3, 250)
            not_ground_mask[inliers] = 0
            xyzr = xyzr[not_ground_mask]

        if self.pnv_prep:
            xyzr = self.pnv_preprocessing(xyzr)
        if self.random_rotation:
            xyzr = self.random_rotate(xyzr)
        if self.random_occlusion:
            xyzr = self.occlude_scan(xyzr)
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            xyzr = scale * xyzr

        return xyzr

    def __getitem__(self, idx):
        drive_id = self.files[idx][0]
        t0 = self.files[idx][1]
        xyz0_th = self.get_pointcloud_tensor(drive_id, t0)
        meta_info = {'drive': drive_id, 't0': t0}

        return (xyz0_th,
                meta_info)


class HeliprTupleDataset(HeliprDataset):
    r"""
    Generate tuples (anchor, positives, negatives) using distance
    Optional other_neg for quadruplet loss. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):
        self.root = root = config.helipr_dir
        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False
        self.pnv_prep = config.pnv_preprocessing
        self.gp_rem = config.gp_rem
        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing HeliprTupleDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = config.helipr_data_split[phase]
        tuple_dir = os.path.join(os.path.dirname(
            __file__), '../../../config/helipr_tuples/')
        
        self.dict_3m = json.load(open(tuple_dir + config.helipr_tp_json, "r"))
        self.dict_20m = json.load(open(tuple_dir + config.helipr_fp_json, "r"))

        self.seq_dict = {}
        for sequence in sequences:
            parent_seq, child_seq = sequence.split('-')
            if parent_seq not in self.seq_dict:
                self.seq_dict[parent_seq] = []
            self.seq_dict[parent_seq].append(child_seq)

        self.fnames_dict = {} # ex. {"KAIST": [fnames1, ...], "DCC": [fnames1, ...]}
        for parent_seq in self.seq_dict: # ex. parent_seq: "KAIST01"
            grandparent_seq = parent_seq[:-2]
            if grandparent_seq not in self.fnames_dict:  # ex. grandparent_seq: "KAIST"
                self.fnames_dict[grandparent_seq] = []
            for child_seq in self.seq_dict[parent_seq]:
                sequence_path = self.root + "/" + parent_seq + "-" + child_seq + '/LiDAR'
                print('sequence_path: ', sequence_path)
                fnames = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))
                self.fnames_dict[grandparent_seq].extend(fnames)
        

        print('fnames_dict: ', self.fnames_dict.keys())
        for k, v in self.fnames_dict.items():
            print(k, len(v))

        self.transform_dict = {} # ex. {"KAIST": [transforms1, ...], "DCC": [transforms1, ...]}
        for parent_seq in self.seq_dict: # ex. parent_seq: "KAIST01"
            grandparent_seq = parent_seq[:-2]
            if grandparent_seq not in self.transform_dict:
                self.transform_dict[grandparent_seq] = []
            for child_seq in self.seq_dict[parent_seq]:
                sequence_path = self.root + "/" + parent_seq + "-" + child_seq + '/scan_poses.csv'
                # sequence_path = os.path.join(self.root, parent_seq, child_seq, 'scan_poses.csv')
                transforms, _ = load_poses_from_csv(sequence_path)
                self.transform_dict[grandparent_seq].extend(transforms)
        for grandparent_seq in self.transform_dict:
            self.transform_dict[grandparent_seq] = np.stack(self.transform_dict[grandparent_seq], axis=0)
        
        print('transform_dict', self.transform_dict.keys())
        for k, v in self.transform_dict.items():
            print(k, len(v))
        # exit()

        self.helipr_seq_lens = {}
        for grandparent_seq in self.fnames_dict.keys():
            self.helipr_seq_lens[grandparent_seq] = len(self.fnames_dict[grandparent_seq])

        print('helipr_seq_lens', self.helipr_seq_lens.keys())
        for k, v in self.helipr_seq_lens.items():
            print(k, v)

        for grandparent_seq, fnames in self.fnames_dict.items():

            for i, fname in enumerate(fnames):
                self.files.append((grandparent_seq, i, 
                                   self.get_positives(grandparent_seq, i), 
                                   self.get_negatives(grandparent_seq, i)))


        # OLD
        # for drive_id in sequences:
        #     sequence_path = self.root + drive_id + '/LiDAR/'
        #     fnames = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))
        #     assert len(
        #         fnames) > 0, f"Make sure that the path {root} has data {drive_id}"
        #     inames = sorted([int(os.path.split(fname)[-1][:-4])
        #                     for fname in fnames])
        #     self.helipr_seq_lens[drive_id] = len(inames)

        #     for query_id, start_time in enumerate(inames):
        #         positives = self.get_positives(drive_id, query_id)
        #         negatives = self.get_negatives(drive_id, query_id)
        #         self.files.append((drive_id, query_id, positives, negatives))



    def get_velodyne_fn(self, parent_sq, query_id): # ex. parent_sq: "KAIST01", query_id: 1000
        if (query_id >= len(self.fnames_dict[parent_sq])):
            print(parent_sq, query_id)
            print(self.fnames_dict[parent_sq])
        return self.fnames_dict[parent_sq][query_id]


    def get_positives(self, parent_sq, index): # ex. parent_sq: "KAIST01", index: 1000
        sq_1 = self.dict_3m[parent_sq] 

        if str(int(index)) in sq_1:
            positives = sq_1[str(int(index))]
        else:
            positives = []

        return positives


    def get_negatives(self, parent_sq, index):  # ex. sq: "KAIST01", index: 1000
        sq_2 = self.dict_20m[parent_sq]

        all_ids = set(np.arange(self.helipr_seq_lens[parent_sq]))
        neg_set_inv = sq_2[str(int(index))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if index in negatives:
            negatives.remove(index)
        return negatives


    def get_other_negative(self, parent_sq, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.helipr_seq_lens[str(parent_sq)])
        neighbour_ids = sel_positive_ids
        for neg in sel_negative_ids:
            neg_postives_files = self.get_positives(parent_sq, neg)
            for pos in neg_postives_files:
                neighbour_ids.append(pos)
        possible_negs = list(set(all_ids) - set(neighbour_ids))
        if query_id in possible_negs:
            possible_negs.remove(query_id)
        assert len(
            possible_negs) > 0, f"No other negatives for sequence {parent_sq} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]


    def __getitem__(self, idx):
        print("[HeliprTupleDataset.__getitem__]: NOT IMPLEMENTED YET")
        return None
        # drive_id, query_id = self.files[idx][0], self.files[idx][1]
        # positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]

        # sel_positive_ids = random.sample(
        #     positive_ids, self.positives_per_query)
        # sel_negative_ids = random.sample(
        #     negative_ids, self.negatives_per_query)
        # positives, negatives, other_neg = [], [], None

        # query_th = self.get_pointcloud_tensor(drive_id, query_id)
        # for sp_id in sel_positive_ids:
        #     positives.append(self.get_pointcloud_tensor(drive_id, sp_id))
        # for sn_id in sel_negative_ids:
        #     negatives.append(self.get_pointcloud_tensor(drive_id, sn_id))

        # meta_info = {'drive': drive_id, 'query_id': query_id}

        # if not self.quadruplet:
        #     return (query_th,
        #             positives,
        #             negatives,
        #             meta_info)
        # else:  # For Quadruplet Loss
        #     other_neg_id = self.get_other_negative(
        #         drive_id, query_id, sel_positive_ids, sel_negative_ids)
        #     other_neg_th = self.get_pointcloud_tensor(drive_id, other_neg_id)
        #     return (query_th,
        #             positives,
        #             negatives,
        #             other_neg_th,
        #             meta_info)

#####################################################################################
# Load poses
#####################################################################################


def load_poses_from_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    if data_poses[0][0] == 'timestamp':
        data_poses = data_poses[1:]
    transforms = []
    positions = []
    for cnt, line in enumerate(data_poses):
        line_f = [float(i) for i in line]
        P = np.vstack((np.reshape(line_f[1:], (3, 4)), [0, 0, 0, 1]))
        transforms.append(P)
        positions.append([P[0, 3], P[1, 3], P[2, 3]])
    return np.asarray(transforms), np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    if data_poses[0][0] == 'timestamp':
        data_poses = data_poses[1:]
    data_poses_ts = np.asarray(
        [float(t)/1e9 for t in np.asarray(data_poses)[:, 0]])
    return data_poses_ts
