import sys
import os
import json
import numpy as np
from termcolor import colored
from tqdm import tqdm
import csv
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.train_config import get_config
from config.eval_config import get_config_eval
cfg = get_config_eval()


def p_dist(pose1, pose2, threshold=3):
    dist = np.linalg.norm(pose1 - pose2)
    if abs(dist) <= threshold:
        return True
    else:
        return False


def t_dist(t1, t2, threshold=10):
    if abs(t1-t2) > threshold:
        return True
    else:
        return False


def get_intra_revisit_indice(basedir: str, sequences: list, 
                             d_thresh: float, t_thresh: float):
    positive_dict = {}
    print(f'd_thresh: {d_thresh}, t_thresh: {t_thresh}')

    for sequence in sequences:
        print(f'processing sequence: {sequence}')
        with open(basedir + sequence + '/scan_poses.csv', newline='') as f:
            reader = csv.reader(f)
            scan_poses = list(reader)
        scan_positions, scan_timestamps = [], []

        for scan_pose in scan_poses:
            scan_position = [float(scan_pose[4]), float(
                scan_pose[8]), float(scan_pose[12])]
            scan_positions.append(np.asarray(scan_position))
            scan_time = int(scan_pose[0])
            scan_timestamps.append(scan_time)
        
        if sequence not in positive_dict:
            positive_dict[sequence] = []

        for t1 in range(len(scan_timestamps)):
            is_revisit = False
            for t2 in range(t1): # Only look at previous timestamps
                if p_dist(scan_positions[t1], scan_positions[t2], d_thresh) & \
                t_dist(scan_timestamps[t1]*1e-9, scan_timestamps[t2]*1e-9, t_thresh):
                    is_revisit = True
                    break
            if is_revisit:
                positive_dict[sequence].append(1)
                print(colored((f't: {t1:5d}, is_revisit: {is_revisit}'), 'green'), end='\r')
            else: 
                positive_dict[sequence].append(0)
                print(colored((f't: {t1:5d}, is_revisit: {is_revisit}'), 'red'), end='\r')
        
        print(f'length of scan: {len(scan_timestamps)}, '
              f'length of positive_dict: {len(positive_dict[sequence])}')
    return positive_dict


def get_inter_revisit_indice(basedir: str, q_sequences: list, db_sequences: list, 
                       d_thresh:float, t_thresh: float):

    positive_dict = {}
    print(f'd_thresh: {d_thresh}, t_thresh: {t_thresh}')

    # end indice for each sequence (0, len(first_seq), len(first_seq)+len(second_seq), ...)
    q_seq_end_indice = [0]

    # Query sequences
    q_scan_positions, q_scan_timestamps = [], []
    for q_seq in q_sequences:
        with open(basedir + q_seq + '/scan_poses.csv', newline='') as f:
            reader = csv.reader(f)
            scan_poses = list(reader)

        print(f'processing query sequence: {q_seq} ... (length: {len(scan_poses)})')
        for scan_pose in scan_poses:
            scan_position = [float(scan_pose[4]), float(
                scan_pose[8]), float(scan_pose[12])]
            q_scan_positions.append(np.asarray(scan_position))
            scan_time = int(scan_pose[0])
            q_scan_timestamps.append(scan_time)

        q_seq_end_indice.append(q_seq_end_indice[-1] + len(scan_poses))

        if q_seq not in positive_dict:
            positive_dict[q_seq] = []

    # Database sequences
    db_scan_positions, db_scan_timestamps = [], []
    for db_seq in db_sequences:
        with open(basedir + db_seq + '/scan_poses.csv', newline='') as f:
            reader = csv.reader(f)
            scan_poses = list(reader)

        print(f'processing database sequence: {db_seq} ... (length: {len(scan_poses)})')
        for scan_pose in scan_poses:
            scan_position = [float(scan_pose[4]), float(
                scan_pose[8]), float(scan_pose[12])]
            db_scan_positions.append(np.asarray(scan_position))
            scan_time = int(scan_pose[0])
            db_scan_timestamps.append(scan_time)


    # save positive dict for each query sequence
    for i, q_seq in enumerate(q_sequences):
        for t1 in range(q_seq_end_indice[i], q_seq_end_indice[i+1]):
            is_revisit = False
            for t2 in range(len(db_scan_timestamps)):
                if p_dist(q_scan_positions[t1], db_scan_positions[t2], d_thresh) & \
                t_dist(q_scan_timestamps[t1]*1e-9, db_scan_timestamps[t2]*1e-9, t_thresh):
                    is_revisit = True
                    break
            if is_revisit:
                positive_dict[q_seq].append(1)
                print(colored((f'[query sequence: {q_seq}] t: {t1:5d}, is_revisit: {is_revisit}'), 'green'), end='\r')
            else: 
                positive_dict[q_seq].append(0)
                print(colored((f'[query sequence: {q_seq}] t: {t1:5d}, is_revisit: {is_revisit}'), 'red'), end='\r')
        
    
    print(f'length of scan: {len(q_scan_timestamps)}, '
            f'length of revisit_list: {len(positive_dict[q_seq])}')
    return positive_dict


def save_intra_revisit_json(basedir: str, sequences: list, output_dir: str,
                             d_thresh: float, t_thresh: float):
    positive_dict = get_intra_revisit_indice(basedir, sequences, d_thresh, t_thresh)
    save_file_name = '{}/is_revisit_intra_D-{}_T-{}.json'.format(
        output_dir, d_thresh, t_thresh)
    with open(save_file_name, 'w') as f:
        json.dump(positive_dict, f)
    print('Saved: ', save_file_name)

    return positive_dict


def save_inter_revisit_json(basedir: str, q_sequences: list, db_sequences: list, output_dir: str,
                             d_thresh: float, t_thresh: float):
    positive_dict = get_inter_revisit_indice(basedir, q_sequences, db_sequences, d_thresh, t_thresh)
    save_file_name = '{}/is_revisit_inter_D-{}_T-{}.json'.format(
        output_dir, d_thresh, t_thresh)
    with open(save_file_name, 'w') as f:
        json.dump(positive_dict, f)
    print('Saved: ', save_file_name)

    return positive_dict

'''
def get_revisit_indice(basedir: str, q_sequences: list, db_sequences: list, 
                       d_thresh:float, t_thresh: float):

    q_scan_positions, q_scan_timestamps = [], []
    for q_seq in q_sequences:
        with open(basedir + q_seq + '/scan_poses.csv', newline='') as f:
            reader = csv.reader(f)
            scan_poses = list(reader)

        print(f'processing query sequence: {q_seq} ... '
              f'(length: {len(scan_poses)})')
        for scan_pose in scan_poses:
            scan_position = [float(scan_pose[4]), float(
                scan_pose[8]), float(scan_pose[12])]
            q_scan_positions.append(np.asarray(scan_position))
            scan_time = int(scan_pose[0])
            q_scan_timestamps.append(scan_time)


    db_scan_positions, db_scan_timestamps = [], []
    for db_seq in db_sequences:
        with open(basedir + db_seq + '/scan_poses.csv', newline='') as f:
            reader = csv.reader(f)
            scan_poses = list(reader)

        print(f'processing database sequence: {db_seq} ... '
              f'(length: {len(scan_poses)})')
        for scan_pose in scan_poses:
            scan_position = [float(scan_pose[4]), float(
                scan_pose[8]), float(scan_pose[12])]
            db_scan_positions.append(np.asarray(scan_position))
            scan_time = int(scan_pose[0])
            db_scan_timestamps.append(scan_time)

    revisit_list = []
    for t1 in range(len(q_scan_timestamps)):
        is_revisit = False
        for t2 in range(len(db_scan_timestamps)):
            if p_dist(q_scan_positions[t1], db_scan_positions[t2], d_thresh) & \
            t_dist(q_scan_timestamps[t1]*1e-9, db_scan_timestamps[t2]*1e-9, t_thresh):
                is_revisit = True
                break
        if is_revisit:
            revisit_list.append(1)
        else: 
            revisit_list.append(0)
        
    
    print(f'length of scan: {len(q_scan_timestamps)}, '
            f'length of revisit_list: {len(revisit_list)}')
    return revisit_list
'''



#####################################################################################
if __name__ == "__main__":

    basedir = cfg.helipr_dir
    query_sequences = [
                'DCC04/Aeva', 'DCC04/Ouster',
                'Riverside04/Aeva', 'Riverside04/Ouster',
                ]
    database_sequences = [
                'DCC05/Aeva', 'DCC05/Avia', 'DCC05/Ouster', 'DCC05/Velodyne',
                ]
    output_dir = os.path.join(os.path.dirname(__file__), '../../config/helipr_tuples/')

    save_intra_revisit_json(basedir, query_sequences, output_dir, 8, 90)
    # save_inter_revisit_json(basedir, query_sequences, database_sequences, output_dir, 8, 90)
