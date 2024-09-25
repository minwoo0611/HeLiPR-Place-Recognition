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

dataset_folder = "/mydata/home/minwoo/Research/PR/HeLiPR-Place-Recognition/"

## =========================================================================================
##                                            SOLiD
## =========================================================================================

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
    """
    Main evaluation function. Evaluates the descriptors on the datasets.

    Returns:
    - dict: Evaluation statistics.
    """
    eval_database_files = ['helipr_validation_db.pickle']
    eval_query_files = ['helipr_validation_query.pickle']

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        location_name = os.path.splitext(database_file)[0]

        with open(os.path.join(dataset_folder, database_file), 'rb') as f:
            database_sets = pickle.load(f)

        with open(os.path.join(dataset_folder, query_file), 'rb') as f:
            query_sets = pickle.load(f)

        stats[location_name] = evaluate_dataset(database_sets, query_sets)

    return stats

def evaluate_dataset(database_sets, query_sets):
    """
    Evaluate the descriptors on a single dataset.

    Parameters:
    - database_sets (list): Database sets.
    - query_sets (list): Query sets.

    Returns:
    - dict: Average recall and one-percent recall.
    """
    num_neighbors = 30
    recall = np.zeros(num_neighbors)
    count = 0
    one_percent_recall = []

    # Compute embeddings
    database_embeddings = [get_latent_vectors(db_set) for db_set in tqdm.tqdm(database_sets, desc='Computing database embeddings')]
    query_embeddings = [get_latent_vectors(q_set) for q_set in tqdm.tqdm(query_sets, desc='Computing query embeddings')]

    # Evaluate pairs
    for i in range(len(database_sets)):
        for j in range(len(query_sets)):
            if (i == 0 and j == 0) or (i == 1 and j == 2):
                pair_recall, pair_opr = get_recall_intra(
                    i, j, database_embeddings, query_embeddings, query_sets, database_sets)
            else:
                pair_recall, pair_opr = get_recall(
                    i, j, database_embeddings, query_embeddings, query_sets, database_sets)
            
            print(f"Pair recall between database set {i} and query set {j}: {pair_recall}")
                
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)

    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    return {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}

def get_latent_vectors(data_set):
    """
    Compute embeddings for a set of point clouds.

    Parameters:
    - data_set (list): List of data entries.

    Returns:
    - np.ndarray: Array of embeddings.
    """
    embeddings = []
    for idx in range(len(data_set)):
        pc_file_path = os.path.join(dataset_folder, data_set[idx]["query"])
        scan = np.fromfile(pc_file_path, dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)])
        points = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)

        # Preprocessing
        points *= 100  # Scaling
        points = remove_closest_points(points, 1)
        points = remove_far_points(points, 100)
        points = down_sampling(points, voxel_size=0.1)

        embedding, _ = get_descriptor(points, 38.6, -38.6, 128, 100)
        embeddings.append(embedding)

    return np.array(embeddings)

def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    """
    Compute recall metrics for inter-sequence place recognition.

    Parameters:
    - m (int): Database index.
    - n (int): Query index.
    - database_vectors (list): Database embeddings.
    - query_vectors (list): Query embeddings.
    - query_sets (list): Query metadata.
    - database_sets (list): Database metadata.

    Returns:
    - tuple: (recall array, one-percent recall value)
    """
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    if np.isnan(database_output).any():
        print('NaN values detected in database embeddings')
    if np.isnan(queries_output).any():
        print('NaN values detected in query embeddings')

    database_nbrs = KDTree(database_output)
    num_neighbors = 30
    recall = np.zeros(num_neighbors)
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    thresholds = np.linspace(0, 1, 250)
    num_thresholds = len(thresholds)

    # Initialize evaluation metrics
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    for i in range(len(queries_output)):
        query_details = query_sets[n][i]
        true_neighbors = query_details.get(m, [])
        if not true_neighbors:
            continue
        num_evaluated += 1

        distances, indices = database_nbrs.query([queries_output[i]], k=num_neighbors)

        for j, idx in enumerate(indices[0]):
            if idx in true_neighbors:
                recall[j:] += 1
                break

        if set(indices[0][:threshold]).intersection(true_neighbors):
            one_percent_retrieved += 1

        # Compute true positives and false positives for thresholds
        for thres_idx, threshold_value in enumerate(thresholds):
            if distances[0][0] < threshold_value:
                if indices[0][0] in true_neighbors:
                    num_true_positive[thres_idx] += 1
                else:
                    num_false_positive[thres_idx] += 1
            else:
                if not true_neighbors:
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1

    # Calculate precision and recall per threshold
    Precisions = np.divide(num_true_positive, num_true_positive + num_false_positive, out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_positive)!=0)
    Recalls = np.divide(num_true_positive, num_true_positive + num_false_negative, out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_negative)!=0)

    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
    recall = (recall / num_evaluated) * 100

    return recall, one_percent_recall

def get_recall_intra(m, n, database_vectors, query_vectors, query_sets, database_sets):
    """
    Compute recall metrics for intra-sequence place recognition.

    Parameters:
    - m (int): Database index.
    - n (int): Query index.
    - database_vectors (list): Database embeddings.
    - query_vectors (list): Query embeddings.
    - query_sets (list): Query metadata.
    - database_sets (list): Database metadata.

    Returns:
    - tuple: (recall array, one-percent recall value)
    """
    database_output = database_vectors[m]

    if np.isnan(database_output).any():
        print('NaN values detected in database embeddings')

    num_neighbors = 30
    recall = np.zeros(num_neighbors)
    one_percent_retrieved = 0
    num_evaluated = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    thresholds = np.linspace(0, 1, 250)
    num_thresholds = len(thresholds)

    # Initialize evaluation metrics
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    # Variables for the trajectory database
    trajectory_db = []
    trajectory_past = []
    trajectory_time = []

    init_time = None

    for i in range(len(database_output)):
        # Extract timestamp from file name
        time_str = os.path.basename(database_sets[m][i]['query']).split('.')[0]
        time = float(time_str) / 1e9

        if init_time is None:
            init_time = time

        trajectory_time.append(time)
        trajectory_past.append(database_output[i])

        # Wait until 90 seconds have passed
        if time - init_time < 90:
            continue

        # Remove old entries (older than 30 seconds)
        while trajectory_time and trajectory_time[0] < time - 30:
            trajectory_db.append(trajectory_past.pop(0))
            trajectory_time.pop(0)

        if len(trajectory_db) < num_neighbors:
            continue

        database_nbrs = KDTree(trajectory_db)
        query_details = query_sets[n][i]
        true_neighbors = query_details.get(m, [])
        if not true_neighbors or min(true_neighbors) >= len(trajectory_db):
            continue

        num_evaluated += 1
        distances, indices = database_nbrs.query([database_output[i]], k=num_neighbors)

        for j, idx in enumerate(indices[0]):
            if idx in true_neighbors:
                recall[j:] += 1
                break

        if set(indices[0][:threshold]).intersection(true_neighbors):
            one_percent_retrieved += 1

        # Compute true positives and false positives for thresholds
        for thres_idx, threshold_value in enumerate(thresholds):
            if distances[0][0] < threshold_value:
                if indices[0][0] in true_neighbors:
                    num_true_positive[thres_idx] += 1
                else:
                    num_false_positive[thres_idx] += 1
            else:
                if not true_neighbors:
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1

    if num_evaluated == 0:
        return np.zeros(num_neighbors), 0.0

    # Calculate precision and recall per threshold
    Precisions = np.divide(num_true_positive, num_true_positive + num_false_positive, out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_positive)!=0)
    Recalls = np.divide(num_true_positive, num_true_positive + num_false_negative, out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_negative)!=0)

    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
    recall = (recall / num_evaluated) * 100

    return recall, one_percent_recall

if __name__ == "__main__":
    stats = evaluate()
    print("Evaluation Statistics:")
    for location, stat in stats.items():
        print(f"Location: {location}")
        print(f"Average One Percent Recall: {stat['ave_one_percent_recall']}")
        print(f"Average Recall: {stat['ave_recall']}")