# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

import os
import sys
import csv
import argparse
import random
import pickle

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import torch
import MinkowskiEngine as ME
import tqdm

np.random.seed(0)
random.seed(0)

# Adjust system path to import local modules
sys.path.append(os.path.dirname(os.getcwd()))

from misc.utils import MinkLocParams
from models.model_factory import model_factory
from datasets.dataset_utils import to_spherical

DEBUG = False

def evaluate(model, device, params, log=False, time_file=None):
    """
    Evaluate the model on all evaluation datasets.

    Parameters:
    - model: The model to evaluate.
    - device: Device to run the evaluation on ('cpu' or 'cuda').
    - params: MinkLocParams object containing configuration parameters.
    - log: If True, log search results.
    - time_file: File to log timing information.

    Returns:
    - stats: Dictionary containing evaluation statistics for each dataset.
    """
    if DEBUG:
        params.eval_database_files = params.eval_database_files[:1]
        params.eval_query_files = params.eval_query_files[:1]

    assert len(params.eval_database_files) == len(params.eval_query_files), \
        "Number of database files and query files must be the same."

    stats = {}
    print("Database files:", params.eval_database_files)
    print("Query files:", params.eval_query_files)

    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, \
            f"Database location: {database_file} does not match query location: {query_file}"

        # Load database and query sets
        db_path = os.path.join(params.dataset_folder, database_file)
        with open(db_path, 'rb') as f:
            database_sets = pickle.load(f)

        query_path = os.path.join(params.dataset_folder, query_file)
        with open(query_path, 'rb') as f:
            query_sets = pickle.load(f)

        # Evaluate on the dataset
        dataset_stats = evaluate_dataset(model, device, params, database_sets, query_sets, log=log, time_file=time_file)
        stats[location_name] = dataset_stats

    return stats


def evaluate_dataset(model, device, params, database_sets, query_sets, log=False, time_file=None):
    """
    Evaluate the model on a single dataset.

    Parameters:
    - model: The model to evaluate.
    - device: Device to run the evaluation on.
    - params: Configuration parameters.
    - database_sets: List of database sets.
    - query_sets: List of query sets.
    - log: If True, log search results.
    - time_file: File to log timing information.

    Returns:
    - stats: Dictionary containing average recall, one-percent recall, and average similarity.
    """
    num_neighbors = 30
    recall = np.zeros(num_neighbors)
    count = 0
    similarity_scores = []
    one_percent_recall_list = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    # Extract embeddings for database sets
    print("Extracting database embeddings...")
    for db_set in tqdm.tqdm(database_sets, desc="Database sets"):
        embeddings = get_latent_vectors(model, db_set, device, params, time_file=time_file)
        database_embeddings.append(embeddings)

    # Extract embeddings for query sets
    print("\nExtracting query embeddings...")
    for q_set in tqdm.tqdm(query_sets, desc="Query sets"):
        embeddings = get_latent_vectors(model, q_set, device, params, time_file=time_file)
        query_embeddings.append(embeddings)

    # Evaluate recall
    for i in range(len(database_sets)):
        for j in range(len(query_sets)):
            # Determine whether to use intra-sequence recall or inter-sequence recall
            if (i == 0 and j == 0) or (i == 1 and j == 2):
                # Intra-sequence evaluation
                pair_recall, pair_similarity, pair_opr = get_recall_intra(
                    i, j, database_embeddings, database_embeddings, query_sets, database_sets, log=log)
            else:
                # Inter-sequence evaluation
                pair_recall, pair_similarity, pair_opr = get_recall(
                    i, j, database_embeddings, query_embeddings, query_sets, database_sets, log=log)

            print(f"Pair recall between database set {i} and query set {j}: {pair_recall}")

            recall += np.array(pair_recall)
            count += 1
            one_percent_recall_list.append(pair_opr)
            similarity_scores.extend(pair_similarity)

    ave_recall = recall / count
    average_similarity = np.mean(similarity_scores)
    ave_one_percent_recall = np.mean(one_percent_recall_list)
    stats = {
        'ave_one_percent_recall': ave_one_percent_recall,
        'ave_recall': ave_recall,
        'average_similarity': average_similarity
    }
    return stats


def pnv_preprocessing(xyz):
    """
    Preprocess point cloud data for PointNetVLAD by downsampling to at most 8192 points.

    Parameters:
    - xyz: Point cloud data as a numpy array.

    Returns:
    - xyz: Downsampled point cloud data.
    """
    voxel_size = 0.3
    while xyz.shape[0] > 8192:
        xyz = downsample_point_cloud(xyz, voxel_size)
        voxel_size += 0.01
    return xyz


def downsample_point_cloud(xyz, voxel_size=0.05):
    """
    Downsample the point cloud using voxel grid filtering.

    Parameters:
    - xyz: Point cloud data as a numpy array.
    - voxel_size: Size of the voxel grid.

    Returns:
    - points: Downsampled point cloud data.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_ds.points)


def load_pc(filename, params):
    """
    Load point cloud data from file.

    Parameters:
    - filename: Path to the point cloud file.
    - params: Configuration parameters.

    Returns:
    - pc: Point cloud data (Nx3 or Nx4 numpy array).
    - pc_s: Transformed point cloud data (e.g., spherical coordinates).
    """
    file_path = os.path.join(params.dataset_folder, filename)

    if params.dataset_name == "HeLiPR":
        pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])

    # Remove intensity channel for models that do not use it
    if params.model_params.version in ['MinkLoc3D', 'MinkLoc3D-S']:
        pc = pc[:, :3]

    # Limit point cloud to max distance
    pc = pc[np.linalg.norm(pc[:, :3], axis=1) < params.max_distance]

    # Convert to spherical coordinates for certain model versions
    if params.model_params.version in ['MinkLoc3D-S', 'MinkLoc3D-SI']:
        pc_s = to_spherical(pc, params.dataset_name)
    else:
        pc_s = pc

    pcs = [pc, pc_s]
    for idx, pc in enumerate(pcs):
        pc_tensor = torch.tensor(pc, dtype=torch.float)
        pad_length = params.num_points - len(pc_tensor)
        if pad_length > 0:
            # Pad with zeros if point cloud has less than num_points points
            pc_tensor = torch.nn.functional.pad(pc_tensor, (0, 0, 0, pad_length), "constant", 0)
        elif pad_length < 0:
            # Truncate if point cloud has more than num_points points
            pc_tensor = pc_tensor[:params.num_points]
        pcs[idx] = pc_tensor

    return pcs[0], pcs[1]


def get_latent_vectors(model, data_set, device, params, time_file=None):
    """
    Compute embeddings for a set of point clouds.

    Parameters:
    - model: The model to compute embeddings with.
    - data_set: Dataset containing point cloud file paths.
    - device: Device to run computations on.
    - params: Configuration parameters.
    - time_file: File to log timing information.

    Returns:
    - embeddings: Numpy array of embeddings for the dataset.
    """
    if DEBUG:
        embeddings = np.random.rand(len(data_set), 256)
        return embeddings

    model.eval()
    embeddings_list = []

    # Check if pointnet features are included
    include_pnt, pnt2s = False, False
    for key in ['pointnet', 'pointnet_cross_attention']:
        if key in params.model_params.combine_params:
            include_pnt = True
            if params.model_params.combine_params[key]['pnt2s']:
                pnt2s = True
    include_pnt2s = include_pnt and pnt2s

    for idx, elem_ndx in enumerate(data_set):
        x, x_s = load_pc(data_set[elem_ndx]["query"], params)
        with torch.no_grad():
            # Prepare inputs based on model version
            if params.model_params.version in ['MinkLoc3D', 'MinkLoc3D-S']:
                coords = ME.utils.sparse_quantize(
                    coordinates=x_s, quantization_size=params.model_params.mink_quantization_size)
                coords_more = ME.utils.sparse_quantize(
                    coordinates=x_s, quantization_size=[0.0001, 0.0001, 0.0001])
                bcoords = ME.utils.batched_coordinates([coords]).to(device)
                bcoords_more = ME.utils.batched_coordinates([coords_more]).to(device)

                # Assign a dummy feature of 1 to each point
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32).to(device)

            elif params.model_params.version in ['MinkLoc3D-I', 'MinkLoc3D-SI']:
                # For models with intensity
                sparse_field = ME.TensorField(
                    features=x_s[:, 3].reshape([-1, 1]),
                    coordinates=ME.utils.batched_coordinates(
                        [x_s[:, :3] / np.array(params.model_params.mink_quantization_size)],
                        dtype=torch.int),
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED).sparse()
                feats = sparse_field.features.to(device)
                bcoords = sparse_field.coordinates.to(device)

            batch = {'coords': bcoords, 'features': feats}
            batch['coords_more'] = bcoords_more if params.model_params.version == 'MinkLoc3D-S' else None

            if include_pnt:
                pnt_coords = x_s if include_pnt2s else x
                batch['pnt_coords'] = pnt_coords.unsqueeze(dim=0).to(device)

            # Compute embedding
            embedding = model(batch, time_file=time_file)

            # Normalize embeddings if required
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        embedding_np = embedding.detach().cpu().numpy()
        embeddings_list.append(embedding_np)

    embeddings = np.vstack(embeddings_list)
    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    """
    Compute recall metrics for inter-sequence place recognition.

    Parameters:
    - m (int): Index of database set.
    - n (int): Index of query set.
    - database_vectors (list of np.ndarray): List containing database embeddings.
    - query_vectors (list of np.ndarray): List containing query embeddings.
    - query_sets (list): List of query metadata dictionaries.
    - database_sets (list): List of database metadata dictionaries.
    - log (bool): If True, log detailed results.

    Returns:
    - recall (np.ndarray): Array of recall values.
    - top1_similarity_score (list): List of top-1 similarity scores.
    - one_percent_recall (float): One percent recall value.
    """
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    database_nbrs = KDTree(database_output)
    num_neighbors = 30
    recall = [0] * num_neighbors
    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    num_evaluated = 0

    for i in range(len(queries_output)):
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]
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

    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
    recall = (np.cumsum(recall) / num_evaluated) * 100

    return recall, top1_similarity_score, one_percent_recall


def get_recall_intra(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    """
    Compute recall metrics for intra-sequence place recognition.

    Parameters:
    - m (int): Index of database set.
    - n (int): Index of query set.
    - database_vectors (list of np.ndarray): List containing database embeddings.
    - query_vectors (list of np.ndarray): List containing query embeddings.
    - query_sets (list): List of query metadata dictionaries.
    - database_sets (list): List of database metadata dictionaries.
    - log (bool): If True, log detailed results.

    Returns:
    - recall (np.ndarray): Array of recall values.
    - top1_similarity_score (list): List of top-1 similarity scores.
    - one_percent_recall (float): One percent recall value.
    """
    database_output = database_vectors[m]

    if np.isnan(database_output).any():
        print('NaN values detected in database embeddings')

    num_neighbors = 30
    recall = [0] * num_neighbors
    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    num_evaluated = 0

    thresholds = np.linspace(0, 1, 250)
    num_thresholds = len(thresholds)
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    trajectory_db = []
    trajectory_past = []
    trajectory_time = []

    init_time = None

    for i in range(len(query_sets[n])):
        time_str = os.path.basename(database_sets[m][i]['query']).split('.')[0]
        time = float(time_str) / 1e9

        if init_time is None:
            init_time = time

        trajectory_time.append(time)
        trajectory_past.append(database_output[i])

        if time - init_time < 90:
            continue

        while trajectory_time and trajectory_time[0] < time - 30:
            trajectory_db.append(trajectory_past.pop(0))
            trajectory_time.pop(0)

        if len(trajectory_db) < num_neighbors:
            continue

        database_nbrs = KDTree(trajectory_db)
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]

        if len(true_neighbors) == 0 or min(true_neighbors) >= len(trajectory_db):
            continue

        num_evaluated += 1

        distances, indices = database_nbrs.query(np.array([database_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                similarity = np.dot(database_output[i], trajectory_db[indices[0][j]])
                top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if set(indices[0][:threshold]).intersection(true_neighbors):
            one_percent_retrieved += 1

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

    if num_evaluated == 0:
        return np.zeros(num_neighbors), [], 0.0

    Precisions = np.divide(num_true_positive, num_true_positive + num_false_positive,
                           out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_positive) != 0)
    Recalls = np.divide(num_true_positive, num_true_positive + num_false_negative,
                        out=np.zeros_like(num_true_positive), where=(num_true_positive + num_false_negative) != 0)

    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
    recall = (np.cumsum(recall) / num_evaluated) * 100

    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    """
    Print evaluation statistics.

    Parameters:
    - stats: Dictionary containing evaluation statistics for each dataset.
    """
    for database_name in stats:
        print(f'Dataset: {database_name}')
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on HeLiPR dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Path to trained model weights')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--log', action='store_true', help='Log search results')

    args = parser.parse_args()

    print(f'Config path: {args.config}')
    print(f'Model config path: {args.model_config}')
    weights = args.weights if args.weights else 'RANDOM WEIGHTS'
    print(f'Weights: {weights}')
    print(f'Debug mode: {args.debug}')
    print(f'Visualization: {args.visualize}')
    print(f'Log search results: {args.log}')
    print('')

    # Set DEBUG flag
    DEBUG = args.debug

    # Load parameters
    params = MinkLocParams(args.config, args.model_config)
    params.print()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')

    # Prepare time logging file
    if args.weights:
        w_list = args.weights.split('/')
        logdir = '/'.join(w_list[:-1])
        epoch = w_list[-1].split('.')[0]
        time_file = os.path.join(logdir, f'{epoch}_time.csv')
    else:
        time_file = 'time.csv'

    with open(time_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Total', 'Pointnet', 'Self-Attention', 'Cross-Attention-Linear', 'Cross-Attention-Dot'])

    # Initialize model
    model = model_factory(params)
    if args.weights:
        assert os.path.exists(args.weights), f'Cannot open network weights: {args.weights}'
        print(f'Loading weights: {args.weights}')
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    # Set evaluation files (Adjust based on your dataset)
    params.eval_database_files = ['helipr_validation_db.pickle']
    params.eval_query_files = ['helipr_validation_query.pickle']

    # Run evaluation
    stats = evaluate(model, device, params, args.log, time_file=time_file)
    print_eval_stats(stats)