# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

import argparse
import os
import sys
import time
import pickle
import numpy as np
import torch
import MinkowskiEngine as ME
import tqdm

from sklearn.neighbors import KDTree

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader

# Define the dataset folder
DATASET_FOLDER = "/PATH/HeLiPR-Place-Recognition/"


def evaluate(model, device, params: TrainingParams, log: bool = False, show_progress: bool = False):
    """
    Run evaluation on all evaluation datasets.

    Parameters:
    - model (nn.Module): The trained PointNetVlad model.
    - device (torch.device): The device to run the model on.
    - params (TrainingParams): Training parameters.
    - log (bool): If True, log detailed results.
    - show_progress (bool): If True, display progress bars.

    Returns:
    - dict: Evaluation statistics for each location.
    """
    # Define evaluation database and query files
    eval_database_files = ['helipr_validation_db.pickle']
    eval_query_files = ['helipr_validation_query.pickle']

    print("Database Files:", eval_database_files)
    print("Query Files:", eval_query_files)

    # Ensure the number of database and query files match
    assert len(eval_database_files) == len(eval_query_files), \
        "Number of database files must match number of query files."

    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from file names
        location_name = database_file.split('_')[0]
        query_location = query_file.split('_')[0]
        assert location_name == query_location, \
            f'Database location: {location_name} does not match query location: {query_location}'

        # Load database and query sets from pickle files
        database_path = os.path.join(DATASET_FOLDER, database_file)
        query_path = os.path.join(DATASET_FOLDER, query_file)

        if not os.path.isfile(database_path):
            print(f"Database file not found: {database_path}")
            continue
        if not os.path.isfile(query_path):
            print(f"Query file not found: {query_path}")
            continue

        with open(database_path, 'rb') as f:
            database_sets = pickle.load(f)

        with open(query_path, 'rb') as f:
            query_sets = pickle.load(f)

        # Evaluate the dataset and collect statistics
        dataset_stats = evaluate_dataset(model, device, params, database_sets, query_sets,
                                        log=log, show_progress=show_progress)
        stats[location_name] = dataset_stats

    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets, log: bool = False,
                    show_progress: bool = False):
    """
    Run evaluation on a single dataset.

    Parameters:
    - model (nn.Module): The trained PointNetVlad model.
    - device (torch.device): The device to run the model on.
    - params (TrainingParams): Training parameters.
    - database_sets (list of dict): List containing database metadata dictionaries.
    - query_sets (list of dict): List containing query metadata dictionaries.
    - log (bool): If True, log detailed results.
    - show_progress (bool): If True, display progress bars.

    Returns:
    - dict: Evaluation statistics including average recall and one-percent recall.
    """
    # Initialize evaluation metrics
    recall = np.zeros(5)  # Number of recall levels (e.g., @1, @2, ..., @5)
    count = 0
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()
    print(f"Number of Queries: {len(query_sets[0])}")
    print(f"Number of Database Sets: {len(database_sets)}")


    # Compute embeddings for database sets
    for set_idx, database_set in enumerate(tqdm.tqdm(database_sets, disable=not show_progress,
                                                   desc='Computing database embeddings')):

        database_embeddings.append(get_latent_vectors(model, database_set, device, params))

    # # Compute embeddings for query sets
    # for set_idx, query_set in enumerate(tqdm.tqdm(query_sets, disable=not show_progress,
    #                                            desc='Computing query embeddings')):

    #     query_embeddings.append(get_latent_vectors(model, query_set, device, params))



    # Iterate over all pairs of database and query embeddings
    for m in range(len(database_embeddings)):
        for n in range(2):
            # Customize intra-sequence evaluation based on indices
            if (m == 0 and n == 0) or (m == 1 and n == 2):
                pair_recall, pair_opr = get_recall_intra(
                    m, n, database_embeddings, query_embeddings, query_sets, database_sets, log=log
                )
            else:
                pair_recall, pair_opr = get_recall(
                    m, n, database_embeddings, query_embeddings, query_sets, database_sets, log=log
                )

            print(f"Pair recall between database set {m} and query set {n}: {pair_recall}")
            
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)

    # Calculate average metrics
    ave_recall = recall / count if count > 0 else np.zeros_like(recall)
    ave_one_percent_recall = np.mean(one_percent_recall) if one_percent_recall else 0.0
    print(f'Average recall: {ave_recall}')

    # Compile statistics
    stats = {
        'ave_one_percent_recall': ave_one_percent_recall,
        'ave_recall': ave_recall
    }
    return stats


def get_latent_vectors(model, data_set, device, params: TrainingParams):
    """
    Compute latent vectors (embeddings) for a given dataset.

    Parameters:
    - model (nn.Module): The trained PointNetVlad model.
    - data_set (dict): A single database or query set containing point cloud file paths.
    - device (torch.device): The device to run the model on.
    - params (TrainingParams): Training parameters.

    Returns:
    - np.ndarray: Array of latent vectors for the dataset.
    """
    # If in debug mode, return random embeddings
    if params.debug:
        embeddings = np.random.rand(len(data_set), 256)
        return embeddings

    pc_loader = PNVPointCloudLoader()
    model.eval()
    embeddings = np.zeros((len(data_set), 256), dtype=np.float32)

    for i, elem_ndx in enumerate(tqdm.tqdm(data_set, desc='Computing latent vectors')):
        pc_file_path = os.path.join(DATASET_FOLDER, data_set[elem_ndx]["query"])
        if not os.path.isfile(pc_file_path):
            print(f"Point cloud file not found: {pc_file_path}")
            continue

        # Load and preprocess the point cloud
        pc = pc_loader(pc_file_path)
        pc = torch.tensor(pc, dtype=torch.float32)

        # Compute the embedding
        embedding = compute_embedding(model, pc, device, params)
        embeddings[i] = embedding

    return embeddings


def compute_embedding(model, pc, device, params: TrainingParams):
    """
    Compute the global descriptor (embedding) for a single point cloud.

    Parameters:
    - model (nn.Module): The trained PointNetVlad model.
    - pc (torch.Tensor): Point cloud tensor.
    - device (torch.device): The device to run the model on.
    - params (TrainingParams): Training parameters.

    Returns:
    - np.ndarray: The computed embedding.
    """
    coords, _ = params.model_params.quantizer(pc)
    with torch.no_grad():
        # Batched coordinates for MinkowskiEngine
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).to(device)
        batch = {'coords': bcoords.to(device), 'features': feats}

        # Compute the global descriptor
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    """
    Compute recall and precision metrics for inter-sequence place recognition.

    Parameters:
    - m (int): Index of the database set.
    - n (int): Index of the query set.
    - database_vectors (list of np.ndarray): List containing database embeddings.
    - query_vectors (list of np.ndarray): List containing query embeddings.
    - query_sets (list of dict): List containing query metadata dictionaries.
    - database_sets (list of dict): List containing database metadata dictionaries.
    - log (bool): If True, log detailed results.

    Returns:
    - tuple:
        - recall (np.ndarray): Array of recall values.
        - precision (np.ndarray): Array of precision values.
        - one_percent_recall (float): One percent recall value.
    """
    # Extract embeddings for the specified database and query sets
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # Initialize KDTree for efficient nearest neighbor search
    database_nbrs = KDTree(database_output)
    num_neighbors = 30
    recall = np.zeros(num_neighbors)
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 1000.0)), 1)
    num_evaluated = 0

    # Initialize arrays to store true/false positives/negatives
    thresholds = np.linspace(0, 1, 250)
    num_thresholds = len(thresholds)
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)


    # Iterate through each query embedding
    for i in range(len(queries_output)):
        query_details = query_sets[n][i]
        true_neighbors = query_details.get(m, [])

        if len(true_neighbors) == 0:
            continue

        num_evaluated += 1

        # Query the nearest neighbors
        distances, indices = database_nbrs.query([queries_output[i]], k=num_neighbors)

        # Check if any of the nearest neighbors are true positives
        for j, idx in enumerate(indices[0]):
            if idx in true_neighbors:
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
                if len(true_neighbors) == 0:
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1


    # Calculate one-percent recall
    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100 if num_evaluated > 0 else 0.0

    # Calculate cumulative recall
    recall = (np.cumsum(recall) / num_evaluated) * 100 if num_evaluated > 0 else np.zeros(num_neighbors)

    # Calculate precision and recall per threshold
    Precisions = np.divide(num_true_positive, num_true_positive + num_false_positive,
                           out=np.zeros_like(num_true_positive),
                           where=(num_true_positive + num_false_positive) != 0)
    Recalls = np.divide(num_true_positive, num_true_positive + num_false_negative,
                        out=np.zeros_like(num_true_positive),
                        where=(num_true_positive + num_false_negative) != 0)


    return recall, one_percent_recall


def get_recall_intra(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    """
    Compute recall metrics for intra-sequence place recognition.

    Parameters:
    - m (int): Index of the database set.
    - n (int): Index of the query set.
    - database_vectors (list of np.ndarray): List containing database embeddings.
    - query_vectors (list of np.ndarray): List containing query embeddings.
    - query_sets (list of dict): List containing query metadata dictionaries.
    - database_sets (list of dict): List containing database metadata dictionaries.
    - log (bool): If True, log detailed results.

    Returns:
    - tuple:
        - recall (np.ndarray): Array of recall values.
        - one_percent_recall (float): One percent recall value.
    """
    database_output = database_vectors[m]

    # Check for NaN values
    if np.isnan(database_output).any():
        print('NaN values detected in database embeddings')

    num_neighbors = 5
    recall = np.zeros(num_neighbors)
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    num_evaluated = 0

    # Initialize thresholds for precision-recall calculations
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

    # Iterate through each query in the intra-sequence
    for i in range(len(query_sets[n])):
        # Extract timestamp from query file name
        query_path = query_sets[n][i]['query']
        time_str = os.path.basename(query_path).split('.')[0]
        try:
            time_val = float(time_str) / 1e9
        except ValueError:
            print(f"Invalid timestamp format: {time_str}")
            continue

        # Initialize start time
        if init_time is None:
            init_time = time_val

        trajectory_time.append(time_val)
        trajectory_past.append(database_output[i])

        # Wait until 90 seconds have passed
        if time_val - init_time < 90:
            continue

        # Remove old entries outside the 30-second window
        while trajectory_time and trajectory_time[0] < time_val - 60:
            trajectory_db.append(trajectory_past.pop(0))
            trajectory_time.pop(0)

        if len(trajectory_db) < num_neighbors:
            continue

        # Initialize KDTree with the current trajectory database
        database_nbrs = KDTree(trajectory_db)
        query_details = query_sets[n][i]
        true_neighbors = query_details.get(m, [])

        # Check if there are true neighbors within the current database
        if not true_neighbors or min(true_neighbors) >= len(trajectory_db):
            continue

        num_evaluated += 1

        # Query the nearest neighbors
        distances, indices = database_nbrs.query([database_output[i]], k=num_neighbors)

        # Check if any of the nearest neighbors are true positives
        for j, idx in enumerate(indices[0]):
            if idx in true_neighbors:
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

    # Calculate one-percent recall
    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100 if num_evaluated > 0 else 0.0

    # Calculate cumulative recall
    recall = (np.cumsum(recall) / num_evaluated) * 100 if num_evaluated > 0 else np.zeros(num_neighbors)

    # Calculate precision and recall per threshold
    Precisions = np.divide(num_true_positive, num_true_positive + num_false_positive,
                           out=np.zeros_like(num_true_positive),
                           where=(num_true_positive + num_false_positive) != 0)
    Recalls = np.divide(num_true_positive, num_true_positive + num_false_negative,
                        out=np.zeros_like(num_true_positive),
                        where=(num_true_positive + num_false_negative) != 0)

    return recall, one_percent_recall


def print_eval_stats(stats):
    """
    Print the evaluation statistics.

    Parameters:
    - stats (dict): Dictionary containing evaluation statistics for each dataset.
    """
    for database_name, stat in stats.items():
        print(f'Dataset: {database_name}')
        print(f"Average Top 1% Recall: {stat['ave_one_percent_recall']:.2f}%")
        print(f"Average Recall @N: {stat['ave_recall']}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PointNetVLAD model on HeLiPR dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Path to trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Enable debug mode with random embeddings')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help='Enable visualization (not implemented)')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true',
                        help='Enable logging of search results')
    parser.set_defaults(log=False)

    args = parser.parse_args()
    print(f'Config path: {args.config}')
    print(f'Model config path: {args.model_config}')
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print(f'Weights: {w}')
    print(f'Debug mode: {args.debug}')
    print(f'Log search results: {args.log}')
    print('')

    # Initialize training parameters
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()  # Ensure TrainingParams has a print method

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Initialize the model using the model factory
    model = model_factory(params.model_params)

    # Load model weights if provided
    if args.weights:
        if not os.path.isfile(args.weights):
            print(f"Model weights not found: {args.weights}")
            sys.exit(1)
        print(f'Loading weights from {args.weights}')
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights provided. Using random weights.')

    # Move the model to the specified device
    model.to(device)

    # Perform evaluation
    stats = evaluate(model, device, params, log=args.log, show_progress=True)
    print_eval_stats(stats)
