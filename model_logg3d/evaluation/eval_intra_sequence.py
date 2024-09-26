from scipy.spatial.distance import cdist
import logging
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.pipelines.pipeline_utils import *
from utils.data_loaders.make_dataloader import *
from utils.misc_utils import *
from utils.data_loaders.helipr.helipr_dataset import load_poses_from_csv, load_timestamps_csv
from sklearn.neighbors import KDTree
import tqdm
__all__ = ['evaluate_intra_sequence']

folder_indices_Roundabout = {
    "Roundabout01-Aeva": 0, 
    "Roundabout01-Avia": 1694,
    "Roundabout01-Ouster": 3389, 
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


def evaluate_intra_sequence(model, cfg):
    
    if "Roundabout" in cfg.eval_seq_q[0]:
        folder_indices = folder_indices_Roundabout
    elif "Town" in cfg.eval_seq_q[0]:
        folder_indices = folder_indices_Town
    elif "Bridge1" in cfg.eval_seq_q[0]:
        folder_indices = folder_indices_Bridge1
    elif "Bridge2" in cfg.eval_seq_q[0]:
        folder_indices = folder_indices
    
    eval_seq = cfg.eval_seq
    cfg.helipr_data_split['test'] = [eval_seq]
    sequence_path = cfg.helipr_dir + eval_seq
    timestamps = load_timestamps_csv(sequence_path + '/trajectory.csv')


    logging.info(f'Evaluating sequence {eval_seq} at {sequence_path}')
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))

    test_loader = make_data_loader(cfg,
                                   cfg.test_phase,
                                   cfg.eval_batch_size,
                                   num_workers=cfg.eval_num_workers,
                                   shuffle=False)

    iterator = test_loader.__iter__()
    logging.info(f'len_dataloader {len(test_loader.dataset)}')

    num_queries = len(timestamps)

    
    # Databases of previously visited/'seen' places.
    seen_descriptors = []

    prep_timer, desc_timer, ret_timer = Timer(), Timer(), Timer()

    start_time = timestamps[0]
    num_neighbors = 30
    recall = [0] * num_neighbors
    num_evaluated = 0

    overlap_score = np.loadtxt(cfg.overlap_path)
    crop_overlap_score = overlap_score[folder_indices[cfg.eval_seq]:folder_indices[cfg.eval_seq]+len(timestamps), folder_indices[cfg.eval_seq]:folder_indices[cfg.eval_seq]+len(timestamps)]
    
    for query_idx in tqdm.tqdm(range(num_queries)):

        input_data = next(iterator)
        prep_timer.tic()
        lidar_pc = input_data[0][0]  # .cpu().detach().numpy()
        if not len(lidar_pc) > 0:
            logging.info(f'Corrupt cloud id: {query_idx}')
            continue
        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
        prep_timer.toc()
        desc_timer.tic()
        output_desc, output_feats = model(input)  # .squeeze()
        desc_timer.toc()
        output_feats = output_feats[0]
        global_descriptor = output_desc.cpu().detach().numpy()

        global_descriptor = np.reshape(global_descriptor, (1, -1))
        query_time = timestamps[query_idx]

        if len(global_descriptor) < 1:
            continue

        seen_descriptors.append(global_descriptor)


        if (query_time - start_time - cfg.skip_time) < 0:
            continue

        # Build retrieval database using entries 30s prior to current query.
        tt = next(x[0] for x in enumerate(timestamps)
                  if x[1] > (query_time - 30))
        db_seen_descriptors = np.copy(seen_descriptors)
        db_seen_descriptors = db_seen_descriptors[:tt+1]
        db_seen_descriptors = db_seen_descriptors.reshape(
            -1, np.shape(global_descriptor)[1])

        # Find top-1 candidate.

        if len(db_seen_descriptors) < 5:
            continue
        ret_timer.tic()
        feat_dists = cdist(global_descriptor, db_seen_descriptors,
                           metric=cfg.eval_feature_distance).reshape(-1)
        ret_timer.toc()

        true_neighbros = np.where(crop_overlap_score[query_idx] > 0.8)

        if len(true_neighbros[0]) == 0:
            continue
        
        num_evaluated += 1
        database_nbrs = KDTree(db_seen_descriptors)
        distances, indices = database_nbrs.query(global_descriptor, k=num_neighbors)
        for i in range(len(indices[0])):
            if np.isin(indices[0][i], true_neighbros):  # Use np.isin to handle numpy array membership check
                recall[i] += 1
                break

    recall = (np.cumsum(recall) / num_evaluated) * 100

    return recall
