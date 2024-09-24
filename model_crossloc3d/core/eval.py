
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle as pkl
import os.path as osp
from sklearn.neighbors import KDTree
import time
from utils import AverageValue, Metrics
    

def get_recall(m, n, database_vectors, query_vectors, query_sets, num_neighbors=30, db_set=None):
    # Original PointNetVLAD code

    if db_set == None:
        pair_dist = None
    else:
        db_set = db_set[m]
        pair_dist = []

    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    thresholds = np.linspace(
        1, 2, int(1000))
    num_thresholds = len(thresholds)

    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        # {'query': path, 'northing': , 'easting': }
        dist = []
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            # from IPython import embed;embed()
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
        # print(distances[0][0])
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

    if num_evaluated == 0:
        return None, None, None, None

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100

    return recall, top1_similarity_score, one_percent_recall, pair_dist


def get_recall_intra(m, n, database_vectors, query_vectors, query_sets, num_neighbors=30, db_set=None):
    # Original PointNetVLAD code

    if db_set == None:
        pair_dist = None
    else:
        db_set = db_set[m]
        pair_dist = []

    database_output = database_vectors[m]
    query_output = query_vectors[n]

    trajectory_db = []
    trajectory_past = []
    trajectory_time = []
    
    recall = [0] * num_neighbors
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)


    num_evaluated = 0
    init_time = 0
    recall = [0] * num_neighbors

    top1_similarity_score = []


    for i in range(len(query_output)):
        time = query_sets[n][i]['query']
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
        
        if len(trajectory_db) < 5:
            continue
        dist = []
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
                if j == 0:
                    similarity = np.dot(
                        database_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1

                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    if num_evaluated == 0:
        return None, None, None, None

  
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100

    return recall, top1_similarity_score, one_percent_recall, pair_dist


def eval(cfg, log, db_data_loader, q_data_loader, task, neighbor=30):

    metrics = AverageValue(Metrics.names())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    task.eval()

    # build database

    all_db_embs = []
    all_db_set = []

    for i in tqdm(range(db_data_loader.dataset.subset_len())):
        db_data_loader.dataset.set_subset(i)

        db_embeddings = []
        db_set = []

        for batch_idx, (meta, data) in enumerate(db_data_loader):

            with torch.no_grad():
                embeddings = task.step(meta, data)
            if cfg.eval_cfg.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            db_embeddings.append(embeddings.detach().cpu().numpy())
            # from IPython import embed;embed()

            if cfg.debug:
                break

        db_embeddings = np.concatenate(db_embeddings, axis=0)
        all_db_embs.append(db_embeddings)
        
    all_q_embs = []
    for i in tqdm(range(q_data_loader.dataset.subset_len())):
        q_data_loader.dataset.set_subset(i)
        q_embeddings = []
        for batch_idx, (meta, data) in enumerate(q_data_loader):

            with torch.no_grad():
                embeddings = task.step(meta, data)
            if cfg.eval_cfg.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            q_embeddings.append(embeddings.detach().cpu().numpy())
            if cfg.debug:
                break
        q_embeddings = np.concatenate(q_embeddings, axis=0)
        all_q_embs.append(q_embeddings)
    # iters = [(i, j) for i in range(len(all_db_embs))
    #          for j in range(len(all_q_embs)) if i != j]

    iters = [(i, j) for i in range(len(all_db_embs))
             for j in range(len(all_q_embs))]

    similarity = []
    dist = []
    recall = np.zeros(neighbor)
    with tqdm(total=len(iters)) as pbar:
        for i, j in iters:
            if (i == 0 and j == 0) or (i == 1 and j == 2):
                pair_recall, pair_similarity, pair_opr, pair_dist = get_recall_intra(
                    i, j, all_db_embs, all_q_embs, q_data_loader.dataset.catalog, num_neighbors=neighbor)
            else:
                pair_recall, pair_similarity, pair_opr, pair_dist = get_recall(
                    i, j, all_db_embs, all_q_embs, q_data_loader.dataset.catalog, num_neighbors=neighbor)
            print("Recall: ", pair_recall)
            if pair_recall is None:
                continue
            _metrics = Metrics.get(pair_opr)
            # from IPython import embed;embed()
            metrics.update(_metrics)
            pbar.update(1)
            recall += np.array(pair_recall)
            for x in pair_similarity:
                similarity.append(x)
            if pair_dist != None:
                for x in pair_dist:
                    dist.append(x)

    avg_recall = recall / len(iters)
    avg_similarity = np.mean(similarity)
    avg_dist = np.mean(dist)
    log.info(
        '====================== EVALUATE RESULTS ======================')
    format_str = '{sample_num:<10} ' + \
        ' '.join(['{%s:<10}' % vn for vn in metrics.value_names])

    title_dict = dict(
        sample_num='Sample'
    )
    title_dict.update({vn: vn for vn in metrics.value_names})

    log.info(format_str.format(**title_dict))

    overall_dict = dict(
        sample_num=len(iters)
    )
    # from IPython import embed;embed()
    overall_dict.update(
        {vn: "%.4f" % metrics.avg(vn) for vn in metrics.value_names})

    log.info(format_str.format(**overall_dict))

    t = 'Avg. similarity: {:.4f} Avg. dist: {:.4f} Avg. recall @N:\n'+str(avg_recall)
    log.info(t.format(avg_similarity, avg_dist))

    return Metrics('Recall@1%', metrics.avg())