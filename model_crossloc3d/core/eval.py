import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KDTree
from utils import AverageValue, Metrics


def get_recall(m, n, database_vectors, query_vectors, query_sets, num_neighbors=30, db_set=None):
    """
    Compute recall metrics for place recognition between database and query vectors.

    Parameters:
    - m (int): Index of the database set
    - n (int): Index of the query set
    - database_vectors (list of np.ndarray): List containing database embeddings
    - query_vectors (list of np.ndarray): List containing query embeddings
    - query_sets (list): List of query metadata dictionaries
    - num_neighbors (int): Number of neighbors to consider in KDTree query
    - db_set (list, optional): Optional database set for additional information

    Returns:
    - recall (np.ndarray): Array of recall values
    - top1_similarity_score (list): List of top-1 similarity scores
    - one_percent_recall (float): One percent recall value
    - pair_dist (list or None): List of pair distances if db_set is provided
    """
    if db_set is not None:
        db_set = db_set[m]
        pair_dist = []
    else:
        pair_dist = None

    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # Build KDTree for database embeddings
    database_nbrs = KDTree(database_output)

    recall = np.zeros(num_neighbors)
    top1_similarity_score = []
    one_percent_retrieved = 0

    thresholds = np.linspace(1, 2, 1000)
    threshold = max(int(round(len(database_output) / 100.0)), 1)
    num_thresholds = len(thresholds)

    # Initialize evaluation metrics
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
    num_evaluated = 0

    for i in range(len(queries_output)):
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]

        if len(true_neighbors) == 0:
            continue

        num_evaluated += 1

        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors)

        # Compute recall
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(set(indices[0][:threshold]).intersection(set(true_neighbors))) > 0:
            one_percent_retrieved += 1

        # Calculate true positives and false positives
        for thres_idx in range(num_thresholds):
            threshold_value = thresholds[thres_idx]
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
        return None, None, None, None

    # Calculate precision and recall
    Precisions = []
    Recalls = []
    for ith_thres in range(num_thresholds):
        nTP = num_true_positive[ith_thres]
        nFP = num_false_positive[ith_thres]
        nTN = num_true_negative[ith_thres]
        nFN = num_false_negative[ith_thres]

        Precision = nTP / (nTP + nFP) if (nTP + nFP) > 0 else 0.0
        Recall = nTP / (nTP + nFN) if (nTP + nFN) > 0 else 0.0

        Precisions.append(Precision)
        Recalls.append(Recall)

    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
    recall = (np.cumsum(recall) / num_evaluated) * 100

    return recall, top1_similarity_score, one_percent_recall, pair_dist


def get_recall_intra(m, n, database_vectors, query_vectors, query_sets, num_neighbors=30, db_set=None):
    """
    Compute recall metrics for intra-sequence place recognition.

    Parameters:
    - m (int): Index of the database set
    - n (int): Index of the query set
    - database_vectors (list of np.ndarray): List containing database embeddings
    - query_vectors (list of np.ndarray): List containing query embeddings
    - query_sets (list): List of query metadata dictionaries
    - num_neighbors (int): Number of neighbors to consider in KDTree query
    - db_set (list, optional): Optional database set for additional information

    Returns:
    - recall (np.ndarray): Array of recall values
    - top1_similarity_score (list): List of top-1 similarity scores
    - one_percent_recall (float): One percent recall value
    - pair_dist (list or None): List of pair distances if db_set is provided
    """
    if db_set is not None:
        db_set = db_set[m]
        pair_dist = []
    else:
        pair_dist = None

    database_output = database_vectors[m]
    query_output = query_vectors[n]

    trajectory_db = []
    trajectory_past = []
    trajectory_time = []

    recall = np.zeros(num_neighbors)
    top1_similarity_score = []
    one_percent_retrieved = 0

    thresholds = np.linspace(1, 2, 1000)
    num_thresholds = len(thresholds)

    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
    num_evaluated = 0

    init_time = 0.0

    for i in range(len(query_output)):
        # Extract timestamp from query path
        time_str = query_sets[n][i]['query'].split('/')[-1].split('.')[0]
        time = float(time_str) / 1e9

        if init_time == 0.0:
            init_time = time

        trajectory_time.append(time)
        trajectory_past.append(database_output[i])

        # Wait until sufficient time has passed
        if time - init_time < 90:
            continue

        # Remove old entries outside the time window
        while len(trajectory_time) > 0 and trajectory_time[0] < time - 30:
            trajectory_db.append(trajectory_past[0])
            trajectory_time.pop(0)
            trajectory_past.pop(0)

        if len(trajectory_db) < 5:
            continue

        database_nbrs = KDTree(trajectory_db)
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]

        if len(true_neighbors) == 0 or min(true_neighbors) >= len(trajectory_db):
            continue

        num_evaluated += 1

        distances, indices = database_nbrs.query(
            np.array([database_output[i]]), k=num_neighbors)

        # Compute recall
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        database_output[i], trajectory_db[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        threshold = max(int(round(len(trajectory_db) / 100.0)), 1)
        if len(set(indices[0][:threshold]).intersection(set(true_neighbors))) > 0:
            one_percent_retrieved += 1

        # Calculate true positives and false positives
        for thres_idx in range(num_thresholds):
            threshold_value = thresholds[thres_idx]
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
        return None, None, None, None

    # Calculate precision and recall
    Precisions = []
    Recalls = []
    for ith_thres in range(num_thresholds):
        nTP = num_true_positive[ith_thres]
        nFP = num_false_positive[ith_thres]
        nTN = num_true_negative[ith_thres]
        nFN = num_false_negative[ith_thres]

        Precision = nTP / (nTP + nFP) if (nTP + nFP) > 0 else 0.0
        Recall = nTP / (nTP + nFN) if (nTP + nFN) > 0 else 0.0

        Precisions.append(Precision)
        Recalls.append(Recall)

    one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
    recall = (np.cumsum(recall) / num_evaluated) * 100

    return recall, top1_similarity_score, one_percent_recall, pair_dist


def eval(cfg, log, db_data_loader, q_data_loader, task, neighbor=30):
    """
    Evaluate the model on the given dataset.

    Parameters:
    - cfg: Configuration object containing evaluation settings
    - log: Logger object for logging evaluation information
    - db_data_loader: DataLoader for the database set
    - q_data_loader: DataLoader for the query set
    - task: Model task to be evaluated
    - neighbor (int): Number of neighbors to consider in KDTree query

    Returns:
    - Metrics object containing evaluation results
    """
    metrics = AverageValue(Metrics.names())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    task.eval()

    # Build database embeddings
    all_db_embs = []
    for i in tqdm(range(db_data_loader.dataset.subset_len()), desc="Building database embeddings"):
        db_data_loader.dataset.set_subset(i)
        db_embeddings = []

        for _, (meta, data) in enumerate(db_data_loader):
            with torch.no_grad():
                embeddings = task.step(meta, data)
            if cfg.eval_cfg.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            db_embeddings.append(embeddings.detach().cpu().numpy())
            if cfg.debug:
                break

        db_embeddings = np.concatenate(db_embeddings, axis=0)
        all_db_embs.append(db_embeddings)

    # Build query embeddings
    all_q_embs = []
    for i in tqdm(range(q_data_loader.dataset.subset_len()), desc="Building query embeddings"):
        q_data_loader.dataset.set_subset(i)
        q_embeddings = []

        for _, (meta, data) in enumerate(q_data_loader):
            with torch.no_grad():
                embeddings = task.step(meta, data)
            if cfg.eval_cfg.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            q_embeddings.append(embeddings.detach().cpu().numpy())
            if cfg.debug:
                break

        q_embeddings = np.concatenate(q_embeddings, axis=0)
        all_q_embs.append(q_embeddings)

    # Prepare pairs for evaluation
    pairs = [(i, j) for i in range(len(all_db_embs))
             for j in range(len(all_q_embs))]

    similarity_scores = []
    distances = []
    total_recall = np.zeros(neighbor)

    with tqdm(total=len(pairs), desc="Evaluating") as pbar:
        for i, j in pairs:
            # Decide whether to perform intra-sequence or inter-sequence evaluation
            if (i == 0 and j == 0) or (i == 1 and j == 2):
                # Intra-sequence evaluation
                pair_recall, pair_similarity, pair_opr, pair_dist = get_recall_intra(
                    i, j, all_db_embs, all_q_embs, q_data_loader.dataset.catalog, num_neighbors=neighbor)
            else:
                # Inter-sequence evaluation
                pair_recall, pair_similarity, pair_opr, pair_dist = get_recall(
                    i, j, all_db_embs, all_q_embs, q_data_loader.dataset.catalog, num_neighbors=neighbor)

            if pair_recall is None:
                continue
            
            print(f"Pair recall between database set {i} and query set {j}: {pair_recall}")
            _metrics = Metrics.get(pair_opr)
            metrics.update(_metrics)
            pbar.update(1)
            total_recall += np.array(pair_recall)
            similarity_scores.extend(pair_similarity)
            if pair_dist is not None:
                distances.extend(pair_dist)

    avg_recall = total_recall / len(pairs)
    avg_similarity = np.mean(similarity_scores)
    avg_distance = np.mean(distances) if distances else None

    # Logging evaluation results
    log.info('====================== EVALUATE RESULTS ======================')
    format_str = '{sample_num:<10} ' + \
        ' '.join(['{%s:<10}' % vn for vn in metrics.value_names])

    title_dict = dict(sample_num='Sample')
    title_dict.update({vn: vn for vn in metrics.value_names})

    log.info(format_str.format(**title_dict))

    overall_dict = dict(sample_num=len(pairs))
    overall_dict.update(
        {vn: "%.4f" % metrics.avg(vn) for vn in metrics.value_names})

    log.info(format_str.format(**overall_dict))

    log.info('Avg. similarity: {:.4f} Avg. distance: {:.4f} Avg. recall @N:\n{}'.format(
        avg_similarity, avg_distance if avg_distance is not None else 0, avg_recall))

    return Metrics('Recall@1%', metrics.avg())
