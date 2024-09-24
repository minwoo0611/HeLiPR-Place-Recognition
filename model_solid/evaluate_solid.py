# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import random
import tqdm
import open3d as o3d

dataset_folder = "/PATH/HeLiPR-Place-Recognition/"

def remove_closest_points(points, thres):
    dists = np.sum(np.square(points[:, :3]), axis=1)
    cloud_out = points[dists > thres*thres]
    return cloud_out

def remove_far_points(points, thres):
    dists = np.sum(np.square(points[:, :3]), axis=1)
    cloud_out = points[dists < thres*thres]
    return cloud_out

def down_sampling(points, voxel_size=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    # Voxel Down-sampling
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # Compute the mean intensity for each voxel
    orig_points_np = np.asarray(pcd.points)
    down_points_np = np.asarray(down_pcd.points)
    # return down_points_np
    return down_points_np


def xy2theta(x, y):
    if (x >= 0 and y >= 0): 
        theta = 180/np.pi * np.arctan(y/x)
    if (x < 0 and y >= 0): 
        theta = 180 - ((180/np.pi) * np.arctan(y/(-x)))
    if (x < 0 and y < 0): 
        theta = 180 + ((180/np.pi) * np.arctan(y/x))
    if ( x >= 0 and y < 0):
        theta = 360 - ((180/np.pi) * np.arctan((-y)/x))
    return theta

def pt2rah(point, gap_ring, gap_sector, gap_height, num_ring, num_sector, num_height, fov_d):
    x = point[0]
    y = point[1]
    z = point[2]
    
    if(x == 0.0):
        x = 0.001  
    if(y == 0.0):
        y = 0.001 

    theta   = xy2theta(x, y) 
    faraway = np.sqrt(x*x + y*y) 
    phi     = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2))) - fov_d

    idx_ring   = np.divmod(faraway, gap_ring)[0]      
    idx_sector = np.divmod(theta, gap_sector)[0]   
    idx_height = np.divmod(phi, gap_height)[0]
    
    if(idx_ring >= num_ring):
        idx_ring = num_ring-1

    if(idx_height >= num_height):
        idx_height = num_height-1

    return int(idx_ring), int(idx_sector), int(idx_height)

## =========================================================================================
##                                            SOLiD
## =========================================================================================

def ptcloud2solid(ptcloud, fov_u, fov_d, num_sector, num_ring, num_height, max_length):
    num_points = ptcloud.shape[0]               
    
    gap_ring = max_length/num_ring            
    gap_sector = 360/num_sector              
    gap_height = ((fov_u-fov_d))/num_height              

    rh_counter = np.zeros([num_ring, num_height])             
    sh_counter = np.zeros([num_sector, num_height])   
    for pt_idx in range(num_points): 
        point = ptcloud[pt_idx, :]
        idx_ring, idx_sector, idx_height = pt2rah(point, gap_ring, gap_sector, gap_height, num_ring, num_sector, num_height, fov_d) 
        try :
            rh_counter[idx_ring, idx_height] = rh_counter[idx_ring, idx_height] + 1     
            sh_counter[idx_sector, idx_height] = sh_counter[idx_sector, idx_height] + 1  
        except:
            pass
            
    ring_matrix = rh_counter    
    sector_matrix = sh_counter
    number_vector = np.sum(ring_matrix, axis=0)
    min_val = number_vector.min()
    max_val = number_vector.max()
    number_vector = (number_vector - min_val) / (max_val - min_val)
        
    r_solid = ring_matrix.dot(number_vector)
    a_solid = sector_matrix.dot(number_vector)
            
    return r_solid, a_solid

def get_descriptor(scan, fov_u, fov_d, num_height, max_length):
    # divide range
    num_ring = 40
    # divide angle
    num_sector = 60

    r_solid, a_solid = ptcloud2solid(scan, fov_u, fov_d, num_sector, num_ring, num_height, max_length)

    return r_solid, a_solid


def evaluate():
    eval_database_files = ['helipr_validation_db.pickle']
    eval_query_files = ['helipr_validation_query.pickle']

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)
        
        temp = evaluate_dataset(database_sets, query_sets)
        stats[location_name] = temp

    return stats


def evaluate_dataset(database_sets, query_sets):
    # Run evaluation on a single dataset
    recall = np.zeros(30)
    count = 0
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    for set in tqdm.tqdm(database_sets, disable=False, desc='Computing database embeddings'):
        database_embeddings.append(get_latent_vectors(set))

    for set in tqdm.tqdm(query_sets, disable=False, desc='Computing query embeddings'):
        query_embeddings.append(get_latent_vectors(set))    



    for i in range(len(database_sets)):
        for j in range(len(query_sets)):
            if (i==0 and j == 0) or (i==1 and j == 2):
                pair_recall, pair_opr = get_recall_intra(i, j, database_embeddings, database_embeddings, query_sets, database_sets)
            else:
                pair_recall, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                               database_sets)
            print(pair_recall)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)


    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}
    return stats


def get_latent_vectors(set):

    embeddings = None
    for i, elem_ndx in enumerate(set):
        pc_file_path = os.path.join(dataset_folder, set[elem_ndx]["query"])
        file_path = os.path.join(pc_file_path)

        lidar_dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
        scan         = np.fromfile(file_path, dtype=lidar_dtype)
        points  = np.stack((scan['x'], scan['y'], scan['z']), axis = -1)

        points *= 100
        query = remove_closest_points(points, 1)
        query = remove_far_points(query, 100)
        query = down_sampling(query, 0.1)
        
        embedding, _ = get_descriptor(query, 38.6, -38.6, 128, 100)

        if embeddings is None:
            embeddings = np.zeros((len(set), embedding.shape[0]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # check nan values
    if np.isnan(database_output).any():
        print('NaN values detected in db embeddings')
    if np.isnan(queries_output).any():
        print('NaN values detected in query embeddings')

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 30
    recall = [0] * num_neighbors

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0

    thresholds = np.linspace(
        0, 1, int(250))
    num_thresholds = len(thresholds)

    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
    

    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break

        if len(list(set(indices[0][0:int(threshold)]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]
            if distances[0][0] < threshold:
                if indices[0][0] in true_neighbors:
                    num_true_positive[thres_idx] += 1
                else:
                    num_false_positive[thres_idx] += 1
            else:
                if len(true_neighbors) == 0:
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1
    
    Precisions, Recalls = [], []
    for ithThres in range(num_thresholds):
        nTruePositive = num_true_positive[ithThres]
        nFalsePositive = num_false_positive[ithThres]
        nTrueNegative = num_true_negative[ithThres]
        nFalseNegative = num_false_negative[ithThres]

        Precision = 0.0
        Recall = 0.0
        if nTruePositive > 0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)
        
        Precisions.append(Precision)
        Recalls.append(Recall)

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100

    return recall, one_percent_recall

def get_recall_intra(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    # check nan values
    if np.isnan(database_output).any():
        print('NaN values detected in db embeddings')

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    trajectory_db = []
    trajectory_past = []
    trajectory_time = []
    num_neighbors = 30
    recall = [0] * num_neighbors
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    init_time = 0
    # query and db are the same
    for i in range(len(database_output)):
        #validation/Roundabout01-Aeva/LiDAR/1689519044659669328.bin
        time = database_sets[m][i]['query']
        time = time.split('/')[-1]
        time = time.split('.')[0]
        
        time = float(time) / 1e9
        if init_time == 0:
            init_time = time
        
        trajectory_time.append(time)
        trajectory_past.append(database_output[i])

        if time - init_time < 90:
            continue

        while len(trajectory_time) > 0 and trajectory_time[0] < time - 30 :
            trajectory_db.append(trajectory_past[0])
            trajectory_time.pop(0)
            trajectory_past.pop(0)
        
        if len(trajectory_db) < num_neighbors:
            continue
        
        database_nbrs = KDTree(trajectory_db)

        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]

        # the length of true neighbors should not be 0 and there should be true neighbor index smaller than the length of trajectory_db
        if len(true_neighbors) == 0:
            continue
        if min(true_neighbors) >= len(trajectory_db):
            continue
        
        num_evaluated += 1

        distances, indices = database_nbrs.query(np.array([database_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, one_percent_recall

if __name__ == "__main__":
    stats = evaluate()


