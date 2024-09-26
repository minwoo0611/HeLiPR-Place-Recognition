import sys
import os
import json
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.train_config import get_config
cfg = get_config()


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

def get_positive_dict(basedir, sequences, output_dir, d_thresh, t_thresh):
    positive_dict = {}
    print('d_thresh: ', d_thresh)
    print('output_dir: ', output_dir)
    print('')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # sequence is a string like 'DCC04/Aeva'
    # seq_dict is a dictionary with keys like 'DCC' and values like ['04', '05']
    # each value is a dictionary with keys like '04' and values like ['Aeva', 'Ouster']
    seq_dict = {}
    prev_indices = 0
    for sequence in sequences:
        parent_seq, child_seq = sequence.split('-') # 'DCC04', 'Aeva'
        grandparent_seq = parent_seq[:-2] # 'DCC'
        if grandparent_seq not in seq_dict:
            seq_dict[grandparent_seq] = {}
        if parent_seq not in seq_dict[grandparent_seq]:
            seq_dict[grandparent_seq][parent_seq] = []
        seq_dict[grandparent_seq][parent_seq].append(child_seq)
    print(seq_dict)

    for grandparent_seq in seq_dict:
        print(grandparent_seq)

        scan_positions, scan_timestamps = [], []
        for parent_seq in seq_dict[grandparent_seq]:
            for child_seq in seq_dict[grandparent_seq][parent_seq]:
                with open( basedir + "/" + parent_seq + "-" + child_seq +  '/trajectory.csv', newline='') as f:
                    reader = csv.reader(f)
                    scan_poses = list(reader)

                for scan_pose in scan_poses:
                    if scan_pose[0] == 'timestamp':
                        continue
                    scan_time = int(scan_pose[0])
                    scan_timestamps.append(scan_time)

        if grandparent_seq not in positive_dict:
            positive_dict[grandparent_seq] = {}

        with open("/mydata/home/oem/minwoo/PR/overlap_matrix_training_overlap_nn.txt", 'rb') as file :
            grandparent_flg = False
            for i, line in enumerate(file):
                if grandparent_flg == False:
                    grandparent_flg = True
                    for j in range(prev_indices):
                        line = next(file)
                    

                if i >= len(scan_timestamps):
                    print("Finished")
                    break
                
                overlaps = np.fromstring(line, dtype=float, sep = ' ')
                
                pos_overlap_list = []
                pos = np.where(overlaps > d_thresh)[0].tolist()
                pos = np.sort(pos)

                pos = pos - prev_indices
                # check pos have any negative value
                if np.any(pos < 0):
                    # pop negative value
                    pos = pos[pos >= 0]

                if np.any(pos >= len(scan_timestamps)):
                    # pop value greater than len(scan_timestamps)
                    pos = pos[pos < len(scan_timestamps)]

                if i not in positive_dict[grandparent_seq]:
                    positive_dict[grandparent_seq][i] = []  

                for p in pos:
                    positive_dict[grandparent_seq][i].append(int(p))

                print(grandparent_seq, i, pos)
        prev_indices = prev_indices + len(scan_timestamps)
    
    for key, sub_dict in positive_dict.items():
        for subkey, subvalue in sub_dict.items():
            print(key, subkey, subvalue)
    # exit(0)?
    
    save_file_name = '{}/positive_sequence_D-{}_T-{}_ouster.json'.format(
        output_dir, d_thresh, t_thresh)
    with open(save_file_name, 'w') as f:
        json.dump(positive_dict, f)
    print('Saved: ', save_file_name)

    return positive_dict


#####################################################################################
if __name__ == "__main__":
    import csv

    basedir = cfg.helipr_dir
    sequences = [
        # TRAIN
        'DCC04-Aeva', 'DCC04-Avia', 'DCC04-Ouster', 'DCC04-Velodyne',
        'DCC05-Aeva', 'DCC05-Avia', 'DCC05-Ouster', 'DCC05-Velodyne',
        'DCC06-Aeva', 'DCC06-Avia', 'DCC06-Ouster', 'DCC06-Velodyne',
        'KAIST04-Aeva', 'KAIST04-Avia', 'KAIST04-Ouster', 'KAIST04-Velodyne',
        'KAIST05-Aeva', 'KAIST05-Avia', 'KAIST05-Ouster', 'KAIST05-Velodyne',
        'KAIST06-Aeva', 'KAIST06-Avia', 'KAIST06-Ouster', 'KAIST06-Velodyne',
        'Riverside04-Aeva', 'Riverside04-Avia', 'Riverside04-Ouster', 'Riverside04-Velodyne',
        'Riverside05-Aeva', 'Riverside05-Avia', 'Riverside05-Ouster', 'Riverside05-Velodyne',
        'Riverside06-Aeva', 'Riverside06-Avia', 'Riverside06-Ouster', 'Riverside06-Velodyne',
             ]
    

    output_dir = os.path.join(os.path.dirname(
        __file__), '../../config/helipr_tuples/')

    t_thresh = 0
    get_positive_dict(basedir, sequences, output_dir, 0.5, t_thresh)
    get_positive_dict(basedir, sequences, output_dir, 0.001, t_thresh)
