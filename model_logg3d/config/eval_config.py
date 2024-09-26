import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')

# Training
# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--train_pipeline', type=str, default='LOGG3D')
eval_arg.add_argument('--eval_pipeline', type=str, default='LOGG3D')
eval_arg.add_argument('--eval_task', type=str, default='intra', choices=['intra', 'inter'])
eval_arg.add_argument('--overlap_path', type=str, default='/PATH/HeLiPR-Place-Recognition/data_overlap/overlap_matrix_validation_Town.txt')
eval_arg.add_argument('--eval_seq', type=str,
                      default='Town01-Ouster')
eval_arg.add_argument('--eval_seq_q', type=str, nargs='+', default=['Town02-Ouster'])
eval_arg.add_argument('--eval_seq_db', type=str, nargs='+', default=['Town01-Ouster'])
eval_arg.add_argument('--checkpoint_name', type=str,
                      default='logg3d_ckpt.pth')
eval_arg.add_argument('--eval_batch_size', type=int, default=1)
eval_arg.add_argument('--eval_num_workers', type=int, default=1)
eval_arg.add_argument("--eval_random_rotation", type=str2bool,
                      default=False, help="If random rotation. ")
eval_arg.add_argument("--eval_random_occlusion", type=str2bool,
                      default=False, help="If random occlusion. ")

eval_arg.add_argument("--revisit_criteria", default=20,
                      type=float, help="in meters")
eval_arg.add_argument("--not_revisit_criteria",
                      default=20, type=float, help="in meters")
eval_arg.add_argument("--skip_time", default=90, type=float, help="in seconds")
eval_arg.add_argument("--cd_thresh_min", default=0.0,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--cd_thresh_max", default=1.0,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--num_thresholds", default=250, type=int,
                      help="Number of thresholds. Number of points on PR curve.")


# Dataset specific configurations
data_arg = add_argument_group('Data')
# KittiDataset #MulRanDataset #HeliprDataset
data_arg.add_argument('--dataset', type=str, default='HeliprDataset')
data_arg.add_argument('--eval_dataset', type=str, default='HeliprDataset')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument("--eval_save_descriptors", type=str2bool, default=True)
data_arg.add_argument("--eval_save_counts", type=str2bool, default=True)
data_arg.add_argument("--eval_save_pr_curve", type=str2bool, default=True)
data_arg.add_argument('--num_points', type=int, default=8192)
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument("--gp_rem", type=str2bool,
                      default=False, help="Remove ground plane.")
data_arg.add_argument('--eval_feature_distance', type=str,
                      default='cosine')  # cosine#euclidean
data_arg.add_argument("--pnv_preprocessing", type=str2bool,
                      default=False, help="Preprocessing in dataloader for PNV.") # PointNetVLAD

data_arg.add_argument('--helipr_dir', type=str,
                      default='PATH/HeLiPR-Place-Recognition/data_validation/', help="Path to the HeLiPR dataset")
data_arg.add_argument("--helipr_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--helipr_data_split', type=dict, default={
    'train': [],
    'val': [],
    'test': []
})

# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=False)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=0.8)
data_arg.add_argument('--max_scale', type=float, default=1.2)


def get_config_eval():
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    cfg = get_config_eval()
    dconfig = vars(cfg)
    print(dconfig)
