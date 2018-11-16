import numpy as np

def get_classification_labels_from_dist_mat(dist_mat, thresh_pos, thresh_neg):
    m, n = dist_mat.shape
    label_mat = np.zeros((m, n))
    num_poses = 0
    num_negs = 0
    pos_pairs = []
    neg_pairs = []
    for i in range(m):
        num_pos = 0
        num_neg = 0
        for j in range(n):
            d = dist_mat[i][j]
            c = classify(d, thresh_pos, thresh_neg)
            if c == 1:
                label_mat[i][j] = 1
                num_pos += 1
                pos_pairs.append((i, j))
            elif c == -1:
                label_mat[i][j] = -1
                num_neg += 1
                neg_pairs.append((i, j))
        num_poses += num_pos
        num_negs += num_neg
    return label_mat, num_poses, num_negs, pos_pairs, neg_pairs


def classify(dist, thresh_pos, thresh_neg):
    if dist <= thresh_pos:
        return 1
    elif dist > thresh_neg:
        return -1
    else:
        return 0
