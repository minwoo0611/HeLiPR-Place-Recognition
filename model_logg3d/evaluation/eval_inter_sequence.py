from scipy.spatial.distance import cdist
import logging
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
from termcolor import colored
from tqdm import tqdm
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.pipelines.pipeline_utils import *
from utils.data_loaders.make_dataloader import *
from utils.misc_utils import *
from utils.data_loaders.helipr.helipr_dataset import load_poses_from_csv, load_timestamps_csv
__all__ = ['evaluate_inter_sequence']

folder_indices_Roundabout = {
    "Roundabout01-Aeva": 0, 
    "Roundabout01-Avia": 1694,
    "Roundabout01-Ouster": 3389, #1696
    "Roundabout01-Velodyne": 5084,
    "Roundabout02-Aeva": 6776,
    "Roundabout02-Avia": 8182,
    "Roundabout02-Ouster": 9587,
    "Roundabout02-Velodyne": 10992,
    "Roundabout03-Aeva": 12397,
    "Roundabout03-Avia": 14135,
    "Roundabout03-Ouster": 15872,
    "Roundabout03-Velodyne": 17608,
}

folder_indices_Town = {
    "Town01-Aeva": 0,
    "Town01-Avia": 1484,
    "Town01-Ouster": 2968,
    "Town01-Velodyne": 4452,
    "Town02-Aeva": 5936,
    "Town02-Avia": 7487,
    "Town02-Ouster": 9038,
    "Town02-Velodyne": 10589,
    "Town03-Aeva": 12141,
    "Town03-Avia": 13827,
    "Town03-Ouster": 15509,
    "Town03-Velodyne": 17193
}

folder_indices_Bridge1 = {
    "Bridge01-Aeva": 0,
    "Bridge01-Avia": 4077,
    "Bridge01-Ouster": 8155,
    "Bridge01-Velodyne": 12231,
    "Bridge04-Aeva": 16298,
    "Bridge04-Avia": 20435,
    "Bridge04-Ouster": 24575,
    "Bridge04-Velodyne": 28713
}

folder_indices_Bridge2 = {
    "Bridge02-Aeva": 0,
    "Bridge02-Avia": 2661,
    "Bridge02-Ouster": 5324,
    "Bridge02-Velodyne": 7988,
    "Bridge03-Aeva": 10650,
    "Bridge03-Avia": 14126,
    "Bridge03-Ouster": 17599,
    "Bridge03-Velodyne": 21071
}


def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, 'ab')
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()
    logging.info(f'Finished saving: {file_name}')


def load_pickle(file_name):
    dbfile = open(file_name, 'rb')
    data_variable = pickle.load(dbfile)
    dbfile.close()
    logging.info(f'Finished loading: {file_name}')
    return data_variable


def parse_sequence(cfg, type):
    assert type in ['query', 'database'], 'type must be either query or database'
    eval_seq = cfg.eval_seq_q if type == 'query' else cfg.eval_seq_db
    sequence_path = [cfg.helipr_dir + seq for seq in eval_seq]
    timestamps = []
    for seq in sequence_path:
        timestamps.append(load_timestamps_csv(seq + '/trajectory.csv'))
    timestamps = np.concatenate(timestamps, axis=0)
    cfg.helipr_data_split['test_q' if type == 'query' else 'test_db'] = eval_seq
    test_loader = make_eval_dataloader(cfg, 'test_q' if type == 'query' else 'test_db')
    iterator = test_loader.__iter__()
    logging.info(f'length of {type} dataloader {len(test_loader.dataset)}')
    return timestamps, iterator


def extract_global_descriptors(model, iterator, cfg, flag):
    descriptors = []
    for idx in tqdm(range(len(iterator))):
        input_data = next(iterator)
        lidar_pc = input_data[0][0]
        
        if not len(lidar_pc) > 0:
            logging.info(f'Corrupt cloud id: {idx}')
            descriptors.append(None)
            continue
        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
        output_desc, _ = model(input)
        global_descriptor = output_desc.cpu().detach().numpy()
        global_descriptor = np.reshape(global_descriptor, (1, -1))

        descriptors.append(global_descriptor)
    return descriptors


def evaluate_inter_sequence(model, cfg, save_dir=None):

    ##### parse query sequences #####
    timestamps_query, iterator_query = parse_sequence(cfg, 'query')

    ##### parse database sequences #####
    timestamps_database, iterator_database = parse_sequence(cfg, 'database')

    ##### Save descriptors and features for Query #####
    descriptors_query = extract_global_descriptors(model, iterator_query, cfg, 1)

    ##### Save descriptors and features for Database #####
    descriptors_database = extract_global_descriptors(model, iterator_database, cfg, 2)

    ##### Delete corrupt clouds #####
    q_idx_to_delete = [idx for idx, desc in enumerate(descriptors_query) if desc is None]    
    timestamps_query = [ts for idx, ts in enumerate(timestamps_query) if idx not in q_idx_to_delete]
    descriptors_query = [desc for idx, desc in enumerate(descriptors_query) if idx not in q_idx_to_delete]

    # set varying thresholds
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))
    num_thresholds = len(thresholds)

    if "Roundabout" in cfg.eval_seq_q[0]:
        folder_indices = folder_indices_Roundabout
    elif "Town" in cfg.eval_seq_q[0]:
        folder_indices = folder_indices_Town
    elif "Bridge1" in cfg.eval_seq_q[0]:
        folder_indices = folder_indices_Bridge1
    elif "Bridge2" in cfg.eval_seq_q[0]:
        folder_indices = folder_indices

    # load txt overlap score
    overlap_score = np.loadtxt(cfg.overlap_path)


    crop_overlap_score = overlap_score[folder_indices[cfg.eval_seq_q[0]]:folder_indices[cfg.eval_seq_q[0]]+len(timestamps_query), folder_indices[cfg.eval_seq_db[0]]:folder_indices[cfg.eval_seq_db[0]]+len(timestamps_database)]
    print(crop_overlap_score.shape)
    database_nbrs = KDTree(np.concatenate(descriptors_database, axis=0))
    num_neighbors = 30
    recall = [0] * num_neighbors
    num_evaluated = 0

    # Find top-1 candidate.
    for q_idx in range(len(timestamps_query)):
        
        true_neighbros = np.where(crop_overlap_score[q_idx] > 0.5)
        if len(true_neighbros[0]) == 0:
            continue
        num_evaluated += 1

        distances, indices = database_nbrs.query(descriptors_query[q_idx], k=num_neighbors)
        for i in range(len(indices[0])):

            if np.isin(indices[0][i], true_neighbros):  # Use np.isin to handle numpy array membership check
                recall[i] += 1
                break

    recall = (np.cumsum(recall) / num_evaluated) * 100

    return recall


if __name__ == "__main__":
    root = '/home/hj/Research/LoGG3D-Net/evaluation/Town03-Ouster_vs_Town01-Ouster'
    num_tp = load_pickle(os.path.join(root, 'num_true_positive.pickle'))
    num_fp = load_pickle(os.path.join(root, 'num_false_positive.pickle'))
    num_tn = load_pickle(os.path.join(root, 'num_true_negative.pickle'))
    num_fn = load_pickle(os.path.join(root, 'num_false_negative.pickle'))
    print(num_tp)
    print(num_fp)
    print(num_tn)
    print(num_fn)