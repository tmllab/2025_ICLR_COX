import os
import time
import copy
import torch
import logging
import numpy as np
import itertools
from torchmetrics import Metric
import matplotlib.pyplot as plt

# # Create a logger object
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)  # Set the logging level

# # Create a console handler
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)

# os.makedirs('./log/', exist_ok=True)
# # Create a file handler
# file_handler = logging.FileHandler('./log/' + str(time.time()) + "logfile.log")
# file_handler.setLevel(logging.INFO)

# # Define a common format for console and file output
# formatter = logging.Formatter("%(levelname)s - %(message)s")
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# # Add handlers to the logger
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)


class Vector(Metric):
    def __init__(self, dim, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("vector", default=torch.zeros(dim), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, vector):
        if isinstance(vector, torch.Tensor):
            vector = vector.detach()
        else:
            vector = torch.tensor(vector).float()
        self.total += vector.shape[0]
        self.vector += vector.sum(0)

    def compute(self):
        return self.vector / self.total
    

class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total
    

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr=0.0):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr + min_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def gather_from_dataset(inputs):
    keys = inputs[0].keys()
    gather_sample = {key: [] for key in keys}
    for sample in inputs:
        for key in gather_sample.keys():
            gather_sample[key].append(sample[key])
    for key, value in gather_sample.items():
        gather_sample[key] = torch.cat(value, dim=0)
    return gather_sample


def break_into_dataset(inputs, num=2):
    keys = inputs.keys()
    sample = {key: [] for key in keys}
    all_sample = [copy.deepcopy(sample) for _ in range(num)]
    for key, value in inputs.items():
        for i in range(num):
            all_sample[i][key] = value.chunk(num)[i]
    return all_sample


def compute_recall(logits):
    sorted_logits = np.sort(-logits, axis=1) # np.sort: from small to large
    pos_logits = np.diag(-logits)
    pos_logits = np.expand_dims(pos_logits, axis=1)
    idx = sorted_logits - pos_logits
    idx = np.where(idx == 0)[1]
    metrics = {}
    try:
        print(1.0 / len(idx))
    except:
        print('logits', logits)
        print('sorted_logits', sorted_logits)
        print('pos_logits', pos_logits)
        print('idx', idx)
    metrics['R1'] = float(np.sum(idx == 0)) * 100 / len(idx)
    metrics['R5'] = float(np.sum(idx < 5)) * 100 / len(idx)
    metrics['R10'] = float(np.sum(idx < 10)) * 100 / len(idx)
    
    return metrics


def get_clip_metrics(logits):
    metrics = {}
    ground_truth = np.arange(len(logits)).reshape(-1, 1)

    ranking = np.argsort(logits)[::-1]
    preds = np.where(ranking == ground_truth)[1]
    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = np.mean(preds < k) * 100

    return metrics



def cal_mAP(similarity_matrix):
    """
    Calculate mean Average Precision (mAP) for two-way retrieval (query-to-gallery and gallery-to-query),
    scaled from 0 to 100.

    Parameters:
    similarity_matrix (np.ndarray): A 2D array where element (i, j) represents 
                                    the similarity score between the i-th query 
                                    and the j-th reference item.

    Returns:
    float: mAP value for two-way retrieval scaled from 0 to 100.
    """
    num_items = similarity_matrix.shape[0]

    def calculate_mAP(sim_matrix, num_queries):
        average_precisions = []

        for i in range(num_queries):
            # Sort similarities and corresponding relevance in descending order
            relevances = sim_matrix[i]
            sorted_indices = np.argsort(relevances)[::-1]
            relevant_items = np.arange(num_queries)  # Assuming ground-truth relevance
            
            # Calculate precision at each rank where a relevant item is found
            true_positive = 0
            cumulative_precision = 0
            for rank, idx in enumerate(sorted_indices, start=1):
                if idx == i:  # Assuming diagonal elements are relevant pairs
                    true_positive += 1
                    cumulative_precision += true_positive / rank

            # Average Precision for this query
            average_precision = cumulative_precision / len(relevant_items)
            average_precisions.append(average_precision)

        # Mean Average Precision (mAP) scaled to 0-100
        return np.mean(average_precisions) * 100

    # Compute mAP for query-to-gallery (e.g., text-to-image)
    mAP_query_to_gallery = calculate_mAP(similarity_matrix, num_items)

    # Compute mAP for gallery-to-query (e.g., image-to-text) by transposing the similarity matrix
    mAP_gallery_to_query = calculate_mAP(similarity_matrix.T, num_items)

    return {
        'mAP_a2b': mAP_query_to_gallery,
        'mAP_b2a': mAP_gallery_to_query
    }




def compute_map(similarity_matrix):
    """
    Compute mean Average Precision (mAP) for two-way retrieval given a similarity matrix.

    Parameters:
    similarity_matrix (numpy.ndarray): The similarity matrix of shape (num_queries, num_candidates).

    Returns:
    float: The mean Average Precision (mAP) in percentage.
    """
    num_queries = similarity_matrix.shape[0]
    outputs = {}

    def one_way_map(similarity_matrix):
        aps = []
        for i in range(num_queries):
            scores = similarity_matrix[i]
            sorted_indices = np.argsort(scores)[::-1]

            num_relevant = 0
            precision_at_k = []

            for rank, index in enumerate(sorted_indices):
                if index == i:
                    num_relevant += 1
                    precision_at_k.append(num_relevant / (rank + 1))
            
            average_precision = np.mean(precision_at_k) if precision_at_k else 0.0
            aps.append(average_precision)
        return np.mean(aps) * 100 

    outputs['mAP_a_to_b'] = one_way_map(similarity_matrix)
    outputs['mAP_b_to_a'] = one_way_map(similarity_matrix.T)
    
    return outputs



def cal_metric_and_print(outputs, print_metric=False):
    keys = list(outputs.keys())
    while 'label' in keys:
        keys.remove('label')
    metrics = {}
    for mod1, mod2 in list(itertools.combinations(keys, 2)):
        outputs1, outputs2 = outputs[mod1].detach(), outputs[mod2].detach()
        logit = torch.softmax((outputs1 / outputs1.norm(p=2, dim=-1, keepdim=True)) @ (outputs2 / outputs2.norm(p=2, dim=-1, keepdim=True)).T, dim=-1).cpu().numpy()
        metrics[f'{mod1}-{mod2}'] = compute_map(logit)

    if print_metric:
        for key, value in metrics.items():
            logging.info(f'modality: {key}:')
            for criteria, result in value.items():
                logging.info(f'{criteria}:\t{result:.6f}')

    return metrics

