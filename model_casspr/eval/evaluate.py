# Evaluation script for PointNetVlad model on HeLiPR dataset

import argparse
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from sklearn.neighbors import KDTree
import tqdm
import pickle
import open3d as o3d

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set base directory and add to system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import custom modules
from loading_pointclouds import load_pc_files  # Ensure this function is defined in loading_pointclouds.py
import models.PointNetVlad as PNV
import config as cfg  # Ensure config.py defines necessary parameters

# Enable cuDNN for optimized performance
cudnn.enabled = True

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    """
    Load the PointNetVlad model, resume from checkpoint, and evaluate on the dataset.
    """
    # Initialize the model with configuration parameters
    model = PNV.PointNetVlad(
        global_feat=True,
        feature_transform=True,
        max_pool=False,
        output_dim=cfg.FEATURE_OUTPUT_DIM,
        num_points=cfg.NUM_POINTS
    ).to(device)

    # Construct the path to the checkpoint file
    resume_filename = os.path.join(cfg.LOG_DIR, "full.ckpt_best.pth")
    print(f"Resuming from {resume_filename}")

    # Load the checkpoint
    if not os.path.isfile(resume_filename):
        print(f"Checkpoint file not found at {resume_filename}")
        sys.exit(1)

    checkpoint = torch.load(resume_filename, map_location=device)
    saved_state_dict = checkpoint.get('state_dict', checkpoint)  # Handle different checkpoint formats

    # Remove 'module.' prefix if present (for models trained with DataParallel)
    new_state_dict = {}
    for key, value in saved_state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load the state dictionary into the model
    model.load_state_dict(new_state_dict)
    print("Model weights loaded successfully.")

    # Wrap the model with DataParallel for multi-GPU support if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")

    # Perform evaluation and print the average one-percent recall
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
    # Load database and query sets from pickle files
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)
    print(f"Number of Database Sets: {len(DATABASE_SETS)}")
    print(f"Number of Query Sets: {len(QUERY_SETS)}")

    # Create results directory if it does not exist
    os.makedirs(cfg.RESULTS_FOLDER, exist_ok=True)

    # Initialize evaluation metrics
    recall = np.zeros(5)  # Number of recall levels (e.g., @1, @2, ..., @5)
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
                    m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS
                )
            else:
                # Inter-sequence evaluation
                pair_recall, pair_similarity, pair_opr = get_recall(
                    m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS
                )
            print(f"Pair Recall: {pair_recall}")
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall_list.append(pair_opr)
            similarity_scores.extend(pair_similarity)

    # Calculate average metrics
    ave_recall = recall / count
    average_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
    ave_one_percent_recall = np.mean(one_percent_recall_list) if one_percent_recall_list else 0.0

    # Save results to the output file
    with open(cfg.OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(f"{ave_recall}\n\n")
        output.write("Average Similarity:\n")
        output.write(f"{average_similarity}\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(f"{ave_one_percent_recall}\n")

    return ave_one_percent_recall


def get_latent_vectors(model, data_set):
    """
    Compute latent vectors (embeddings) for a given dataset.

    Parameters:
    - model (nn.Module): The trained PointNetVlad model.
    - data_set (dict): A single database or query set containing point cloud file paths.

    Returns:
    - np.ndarray: Array of latent vectors for the dataset.
    """
    model.eval()
    embeddings_list = []

    # Generate indices for batching
    train_file_idxs = np.arange(len(data_set.keys()))

    # Calculate the number of files per batch
    batch_num = cfg.EVAL_BATCH_SIZE * (
        1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY
    )
    q_output = []

    # Process data in batches
    for q_index in range(len(train_file_idxs) // batch_num):
        file_indices = train_file_idxs[q_index * batch_num:(q_index + 1) * batch_num]
        file_names = [data_set[index]["query"] for index in file_indices]
        queries = load_pc_files(file_names)  # Ensure load_pc_files returns a numpy array

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)  # Add channel dimension if required
            feed_tensor = feed_tensor.to(device)
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)
        q_output.append(out)

    # Concatenate batch outputs
    q_output = np.array(q_output)
    if q_output.size != 0:
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # Handle edge case where number of files is not a multiple of batch_num
    index_edge = (len(train_file_idxs) // batch_num) * batch_num
    if index_edge < len(data_set.keys()):
        file_indices = train_file_idxs[index_edge:]
        file_names = [data_set[index]["query"] for index in file_indices]
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out = model(feed_tensor)

        output = out.detach().cpu().numpy()
        output = np.squeeze(output)
        if q_output.size != 0:
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS, log=False):
    """
    Compute recall metrics for inter-sequence place recognition.

    Parameters:
    - m (int): Index of the database set.
    - n (int): Index of the query set.
    - DATABASE_VECTORS (list of np.ndarray): List containing database embeddings.
    - QUERY_VECTORS (list of np.ndarray): List containing query embeddings.
    - QUERY_SETS (list of dict): List containing query metadata dictionaries.
    - log (bool): If True, log detailed results.

    Returns:
    - tuple: (recall array, list of top-1 similarity scores, one-percent recall value)
    """
    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    # Initialize KDTree for efficient nearest neighbor search
    database_nbrs = KDTree(database_output)
    num_neighbors = 30
    recall = [0] * num_neighbors
    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    num_evaluated = 0

    for i in range(len(queries_output)):
        query_details = QUERY_SETS[n][i]
        true_neighbors = query_details.get(m, [])
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if set(indices[0][:threshold]).intersection(true_neighbors):
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100 if num_evaluated > 0 else 0.0
    recall = (np.cumsum(recall) / num_evaluated) * 100 if num_evaluated > 0 else np.zeros(num_neighbors)

    return recall, top1_similarity_score, one_percent_recall


def get_recall_intra(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS, database_sets, log=False):
    """
    Compute recall metrics for intra-sequence place recognition.

    Parameters:
    - m (int): Index of the database set.
    - n (int): Index of the query set.
    - DATABASE_VECTORS (list of np.ndarray): List containing database embeddings.
    - QUERY_VECTORS (list of np.ndarray): List containing query embeddings.
    - QUERY_SETS (list of dict): List containing query metadata dictionaries.
    - database_sets (list of dict): List containing database metadata dictionaries.
    - log (bool): If True, log detailed results.

    Returns:
    - tuple: (recall array, list of top-1 similarity scores, one-percent recall value)
    """
    database_output = DATABASE_VECTORS[m]

    if np.isnan(database_output).any():
        print('NaN values detected in database embeddings')

    num_neighbors = 30
    recall = [0] * num_neighbors
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

    for i in range(len(QUERY_SETS[n])):
        # Extract timestamp from query file name
        query_path = QUERY_SETS[n][i]['query']
        time_str = os.path.basename(query_path).split('.')[0]
        try:
            time = float(time_str) / 1e9
        except ValueError:
            print(f"Invalid timestamp format: {time_str}")
            continue

        # Initialize start time
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
        query_details = QUERY_SETS[n][i]
        true_neighbors = query_details.get(m, [])

        # Check if there are true neighbors within the current database
        if not true_neighbors or min(true_neighbors) >= len(trajectory_db):
            continue

        num_evaluated += 1

        # Query the nearest neighbors
        distances, indices = database_nbrs.query(np.array([database_output[i]]), k=num_neighbors)

        # Check if any of the nearest neighbors are true positives
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                if j == 0:
                    similarity = np.dot(QUERY_VECTORS[n][i], trajectory_db[indices[0][j]])
                    top1_similarity_score.append(similarity)
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

    print(f"Recall @1: {recall[0]}, Evaluated Queries: {num_evaluated}")
    return recall, top1_similarity_score, one_percent_recall


def get_sets_dict(pickle_file):
    """
    Load sets from a pickle file.

    Parameters:
    - pickle_file (str): Path to the pickle file.

    Returns:
    - list of dict: Loaded sets.
    """
    if not os.path.isfile(pickle_file):
        print(f"Pickle file not found: {pickle_file}")
        sys.exit(1)

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

    FLAGS = parser.parse_args()

    # Update configuration parameters based on arguments
    cfg.NUM_POINTS = 8192
    cfg.FEATURE_OUTPUT_DIM = 256
    cfg.EVAL_POSITIVES_PER_QUERY = FLAGS.positives_per_query
    cfg.EVAL_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
    cfg.EVAL_BATCH_SIZE = FLAGS.eval_batch_size
    cfg.DECAY_STEP = FLAGS.decay_step
    cfg.DECAY_RATE = FLAGS.decay_rate
    cfg.RESULTS_FOLDER = FLAGS.results_dir

    # Set evaluation file paths
    cfg.EVAL_DATABASE_FILE = '../PR/helipr_Bridge0203_validation_crossloc_5_db.pickle'
    cfg.EVAL_QUERY_FILE = '../PR/helipr_Bridge0203_validation_crossloc_5_query.pickle'

    # Set logging directories and output file
    cfg.LOG_DIR = FLAGS.log_dir
    cfg.OUTPUT_FILE = os.path.join(cfg.RESULTS_FOLDER, 'results.txt')
    cfg.MODEL_FILENAME = "model.ckpt"

    # Ensure the results directory exists
    os.makedirs(cfg.RESULTS_FOLDER, exist_ok=True)

    # Start evaluation
    evaluate()