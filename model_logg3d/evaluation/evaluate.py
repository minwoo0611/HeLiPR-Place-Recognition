import os
import sys
import torch
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.misc_utils import log_config
from evaluation.eval_intra_sequence import evaluate_intra_sequence
from evaluation.eval_inter_sequence import evaluate_inter_sequence
from datetime import datetime

log_save_dir = os.path.join("logs", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(log_save_dir, exist_ok=True)
info_file_handler = logging.FileHandler(os.path.join(log_save_dir, 'info.log'))
console_handler = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[info_file_handler, console_handler])
logging.basicConfig(level=logging.INFO, format="")


if __name__ == "__main__":
    from models.pipeline_factory import get_pipeline
    from config.eval_config import get_config_eval

    cfg = get_config_eval()

    # Get model
    model = get_pipeline(cfg.eval_pipeline)

    ckpt_path = os.path.join(os.path.dirname(__file__), '../', '../data_ckpt/')
    ckpt_path = str(ckpt_path) + cfg.checkpoint_name
    print('Loading checkpoint from: ', ckpt_path)
    logging.info('\n' + ' '.join([sys.executable] + sys.argv))
    log_config(cfg, logging)

    checkpoint = torch.load(ckpt_path)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()

    logging.info('Task: ' + str(cfg.eval_task))
    logging.info('Checkpoint Name: ' + str(cfg.checkpoint_name))
    
    if cfg.eval_task == 'intra':
        eval_recall = evaluate_intra_sequence(model, cfg)
        logging.info(f'Evaluated Sequence: {cfg.eval_seq}')
    elif cfg.eval_task == 'inter':
        eval_recall = evaluate_inter_sequence(model, cfg, log_save_dir)
        logging.info(f'Evaluated Sequence ==> '
                     f'Query: {cfg.eval_seq_q}, '
                     f'Database: {cfg.eval_seq_db}')

    logging.info(
        '\n' + '******************* Evaluation Complete *******************')
    logging.info('Recall: ' + str(eval_recall))