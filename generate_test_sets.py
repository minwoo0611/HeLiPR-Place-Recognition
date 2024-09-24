# Code taken from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, overlap_matrix, db_folder, query_folder):
    test_sets = []
    database_sets = []
    bin_idx = 0
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(
            base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder+folder + \
            pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():            
            if folder in db_folder:
                database[len(database.keys())] = {
                    'query': row['file'], 'x': row['x'], 'y': row['y'], 'idx' : bin_idx}
                
            if folder in query_folder:
                test[len(test.keys())] = {'query': row['file'], 'x': row['x'], 'y': row['y'], 'idx' : bin_idx}
            
            bin_idx = bin_idx + 1

        if folder in db_folder:
            database_sets.append(database)
        
        if folder in query_folder:
            test_sets.append(test)

    overlap = np.loadtxt(overlap_matrix)   

    for k in range(len(database_sets)):
        for i in range(len(test_sets)):
            for key in range(len(test_sets[i].keys())):
                index = []
                for j in range(len(database_sets[k])):
                    if(float(overlap[test_sets[i][key]['idx'], database_sets[k][j]['idx']]) > 0.5):
                        # common index in index and index_dist
                        index.append(j)

                test_sets[i][key][k] = index

    for i in range(len(database_sets)):
        for key in range(len(database_sets[i].keys())):
            del database_sets[i][key]['idx']
    for i in range(len(test_sets)):
        for key in range(len(test_sets[i].keys())):
            del test_sets[i][key]['idx']

    output_to_file(database_sets, base_path+'helipr_validation_db.pickle')
    output_to_file(test_sets, base_path+'helipr_validation_query.pickle')

base_path = "/mydata/home/minwoo/Research/PR/HeLiPR-Place-Recognition/"
runs_folder = "data_validation/"
pcd_folder = "/LiDAR/"
overlap_matrix = base_path + "data_overlap/" + "overlap_matrix_validation_Town.txt"

location = "Town"

db_folder = ["Town01-Ouster", "Town01-Aeva"]
query_folder = [ "Town01-Aeva", "Town01-Avia", "Town01-Ouster", "Town01-Velodyne"]

folders = []
all_folders = sorted(os.listdir(
    os.path.join(base_path, runs_folder)))

for folder in all_folders:
    if location in folder:
        folders.append(folder)

construct_query_and_database_sets(base_path, runs_folder, folders, pcd_folder, "trajectory.csv", overlap_matrix, db_folder, query_folder)