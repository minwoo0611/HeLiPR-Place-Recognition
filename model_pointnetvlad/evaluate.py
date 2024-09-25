# Evaluation code adapted from PointNetVlad: https://github.com/mikacuy/pointnetvlad

import os
import sys
import argparse
import csv
import random
import pickle
import math

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from sklearn.neighbors import KDTree
import tqdm
import open3d as o3d

# Set random seeds for reproducibility
np.random.seed(0)
random.seed(0)

# Adjust system path to include necessary modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from loading_pointclouds import load_pc_files
import models.PointNetVlad as PNV
from tensorboardX import SummaryWriter
import loss.pointnetvlad_loss
import config as cfg

cudnn.enabled = True

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    """
    Load the PointNetVlad model, resume from checkpoint, and evaluate on the dataset.
    """
    # Initialize the model
    model = PNV.PointNetVlad(
        global_feat=True,
        feature_transform=True,
        max_pool=False,
        output_dim=cfg.FEATURE_OUTPUT_DIM,
        num_points=cfg.NUM_POINTS
    ).to(device)

    # Load model weights
    resume_filename = cfg.MODEL_FILENAME
    print(f"Resuming from {resume_filename}")
    checkpoint = torch.load(resume_filename, map_location=device)
    saved_state_dict = checkpoint['state_dict']
    
    # Remove 'module.' prefix if present (for models trained with DataParallel)
    new_state_dict = {}
    for key, value in saved_state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    
    # Load the state dict into the model
    model.load_state_dict(new_state_dict)
    model = nn.DataParallel(model)
    
    # Perform evaluation
    average_one_percent_recall = evaluate_model(model)
    print(f"Average Top 1% Recall: {average_one_percent_recall:.2f}")


def evaluate_model(model):
    """
    Evaluate the model on the specified database and query sets.

    Parameters:
    - model (nn.Module): The trained PointNetVlad model.
    
    Returns:
    - float: The average top 1% recall across all evaluations.
    """
    # Load database and query sets
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)
    
    print(f"Number of Database Sets: {len(DATABASE_SETS)}")
    print(f"Number of Query Sets: {len(QUERY_SETS)}")
    
    # Initialize evaluation metrics
    recall = np.zeros(30)  # Adjust the number of recall levels if needed
    count = 0
    similarity_scores = []
    one_percent_recall_list = []
    
    # Compute embeddings for database and query sets
    DATABASE_VECTORS = []
    QUERY_VECTORS = []
    
    print("Extracting database embeddings...")
    for i in tqdm.tqdm(range(len(DATABASE_SETS)), desc="Database Sets"):
        DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))
        
    print("\nExtracting query embeddings...")
    for j in tqdm.tqdm(range(len(QUERY_SETS)), desc="Query Sets"):
        QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))
    
    # Evaluate recall metrics for each pair of database and query sets
    for m in range(len(DATABASE_VECTORS)):
        for n in range(len(QUERY_VECTORS)):
            if (m == 0 and n == 0) or (m == 1 and n == 2):
                # Intra-sequence evaluation
                pair_recall, pair_similarity, pair_opr = get_recall_intra(
                    m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS, DATABASE_SETS
                )
            else:
                # Inter-sequence evaluation
                pair_recall, pair_similarity, pair_opr = get_recall(
                    m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS, DATABASE_SETS
                )
            
            print(f"Pair recall between database set {m} and query set {n}: {pair_recall}")
            
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall_list.append(pair_opr)
            similarity_scores.extend(pair_similarity)
    
    # Calculate average metrics
    ave_recall = recall / count
    average_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
    ave_one_percent_recall = np.mean(one_percent_recall_list) if one_percent_recall_list else 0.0
    
    return ave_one_percent_recall


def get_latent_vectors(model, dict_to_process):

    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.EVAL_BATCH_SIZE * \
        (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    # print(q_output.shape)
    return q_output


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    """
    Compute recall metrics for inter-sequence place recognition.

    Parameters:
    - m (int): Index of the database set.
    - n (int): Index of the query set.
    - database_vectors (list of np.ndarray): List containing database embeddings.
    - query_vectors (list of np.ndarray): List containing query embeddings.
    - query_sets (list of dict): List containing query metadata dictionaries.
    - database_sets (list of dict): List containing database metadata dictionaries.
    
    Returns:
    - tuple: (recall array, list of top-1 similarity scores, one-percent recall value)
    """
    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    
    # Initialize KDTree for efficient nearest neighbor search
    database_nbrs = KDTree(database_output)
    
    num_neighbors = 30
    recall = np.zeros(num_neighbors)
    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    
    num_evaluated = 0
    
    # Initialize arrays to store true/false positives/negatives
    thresholds = np.linspace(0, 1, 250)
    num_thresholds = len(thresholds)
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
        
        # Query the nearest neighbors
        distances, indices = database_nbrs.query([queries_output[i]], k=num_neighbors)
        
        # Check if any of the nearest neighbors are true positives
        for j, idx in enumerate(indices[0]):
            if idx in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[idx])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
        
        # Check for one-percent recall
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
    Precisions = np.divide(num_true_positive, num_true_positive + num_false_positive,
                           out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_positive) != 0)
    Recalls = np.divide(num_true_positive, num_true_positive + num_false_negative,
                        out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_negative) != 0)
    
    # Calculate one-percent recall
    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100 if num_evaluated > 0 else 0.0
    recall = (np.cumsum(recall) / num_evaluated) * 100 if num_evaluated > 0 else np.zeros(num_neighbors)
    
    return recall, top1_similarity_score, one_percent_recall


def get_recall_intra(m, n, database_vectors, query_vectors, query_sets, database_sets):
    """
    Compute recall metrics for intra-sequence place recognition.

    Parameters:
    - m (int): Index of the database set.
    - n (int): Index of the query set.
    - database_vectors (list of np.ndarray): List containing database embeddings.
    - query_vectors (list of np.ndarray): List containing query embeddings.
    - query_sets (list of dict): List containing query metadata dictionaries.
    - database_sets (list of dict): List containing database metadata dictionaries.
    
    Returns:
    - tuple: (recall array, list of top-1 similarity scores, one-percent recall value)
    """
    database_output = database_vectors[m]
    
    if np.isnan(database_output).any():
        print('NaN values detected in database embeddings')
    
    num_neighbors = 30
    recall = np.zeros(num_neighbors)
    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    num_evaluated = 0
    
    # Initialize arrays to store true/false positives/negatives
    thresholds = np.linspace(0, 1, 250)
    num_thresholds = len(thresholds)
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
    
    # Initialize variables for trajectory-based evaluation
    trajectory_db = []
    trajectory_past = []
    trajectory_time = []
    init_time = None
    
    for i in range(len(query_sets[m])):
        # Extract timestamp from query file name
        time_str = os.path.basename(database_sets[m][i]['query']).split('.')[0]
        time = float(time_str) / 1e9
        
        if init_time is None:
            init_time = time
        
        trajectory_time.append(time)
        trajectory_past.append(database_output[i])
        
        # Wait until 90 seconds have passed
        if time - init_time < 90:
            continue
        
        # Remove old entries outside the 30-second window
        while trajectory_time and trajectory_time[0] < time - 30:
            trajectory_db.append(trajectory_past.pop(0))
            trajectory_time.pop(0)
        
        if len(trajectory_db) < num_neighbors:
            continue
        
        # Initialize KDTree with the current trajectory database
        database_nbrs = KDTree(trajectory_db)
        query_details = query_sets[n][i]
        true_neighbors = query_details.get(m, [])
        
        if not true_neighbors or min(true_neighbors) >= len(trajectory_db):
            continue
        
        num_evaluated += 1
        
        # Query the nearest neighbors
        distances, indices = database_nbrs.query([database_output[i]], k=num_neighbors)
        
        # Check if any of the nearest neighbors are true positives
        for j, idx in enumerate(indices[0]):
            if idx in true_neighbors:
                similarity = np.dot(query_vectors[n][i], database_output[idx])
                top1_similarity_score.append(similarity)
                recall[j] += 1
                break
        
        # Check for one-percent recall
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
    Precisions = np.divide(num_true_positive, num_true_positive + num_false_positive,
                           out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_positive) != 0)
    Recalls = np.divide(num_true_positive, num_true_positive + num_false_negative,
                        out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_negative) != 0)
    
    # Calculate one-percent recall
    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100 if num_evaluated > 0 else 0.0
    recall = (np.cumsum(recall) / num_evaluated) * 100 if num_evaluated > 0 else np.zeros(num_neighbors)
    
    return recall, one_percent_recall


def get_sets_dict(pickle_file):
    """
    Load sets from a pickle file.

    Parameters:
    - pickle_file (str): Path to the pickle file.
    
    Returns:
    - list of dict: Loaded sets.
    """
    with open(pickle_file, 'rb') as f:
        sets = pickle.load(f)
    return sets




if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Evaluate PointNetVlad model on HeLiPR dataset')
    parser.add_argument('--positives_per_query', type=int, default=4,
                        help='Number of positive samples per query [default: 4]')
    parser.add_argument('--negatives_per_query', type=int, default=12,
                        help='Number of negative samples per query [default: 12]')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='Batch size during evaluation [default: 1]')
    parser.add_argument('--dimension', type=int, default=256,
                        help='Dimension of the output feature vectors [default: 256]')
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for learning rate decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for learning rate decay [default: 0.7]')
    parser.add_argument('--results_dir', default='results/',
                        help='Directory to save results [default: results/]')
    parser.add_argument('--dataset_folder', default='../PR/',
                        help='Folder containing the PointNetVlad dataset [default: ../PR/]')
    parser.add_argument('--log_dir', default='log/',
                        help='Directory containing the model logs [default: log/]')
    
    args = parser.parse_args()
    
    # Update configuration parameters based on arguments
    cfg.NUM_POINTS = 8192
    cfg.FEATURE_OUTPUT_DIM = args.dimension
    cfg.EVAL_POSITIVES_PER_QUERY = args.positives_per_query
    cfg.EVAL_NEGATIVES_PER_QUERY = args.negatives_per_query

    cfg.EVAL_DATABASE_FILE = '../helipr_validation_db.pickle'
    cfg.EVAL_QUERY_FILE = '../helipr_validation_query.pickle'
    cfg.MODEL_FILENAME = "../data_ckpt/pointnetvlad_ckpt.pth"
    cfg.NORMALIZE_EMBEDDINGS = True  # Ensure embeddings are normalized if required

    # Start evaluation
    evaluate()

