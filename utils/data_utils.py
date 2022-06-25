import os
import csv
from scipy import sparse
import pandas as pd
import numpy as np

def load_data(dataset):
    pro_dir = os.path.join('data', dataset)
    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as _r:
        for line in _r:
            unique_sid.append(line.strip())
    n_items = len(unique_sid)

    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items)

    vad_data_tr, vad_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'validation_tr.csv'),
        os.path.join(pro_dir, 'validation_te.csv'),
        n_items)

    tst_data_tr, tst_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'test_tr.csv'),
        os.path.join(pro_dir, 'test_te.csv'),
        n_items)

    assert n_items == train_data.shape[1]
    assert n_items == vad_data_tr.shape[1]
    assert n_items == vad_data_te.shape[1]
    assert n_items == tst_data_tr.shape[1]
    assert n_items == tst_data_te.shape[1]

    return (n_items, train_data, vad_data_tr, vad_data_te,
            tst_data_tr, tst_data_te)


def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def load_item_cate(data_dir, num_items):
    assert 'alishop' in data_dir
    data_dir = os.path.join(data_dir, 'pro_sg')

    hash_to_sid = {}
    with open(os.path.join(data_dir, 'unique_sid.txt')) as fin:
        for i, line in enumerate(fin):
            hash_to_sid[int(line)] = i
    assert num_items == len(hash_to_sid)

    hash_to_cid = {}
    with open(os.path.join(data_dir, 'item_cate.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for item, cate in reader:
            item, cate = int(item), int(cate)
            if item not in hash_to_sid:
                continue
            assert item in hash_to_sid
            if cate not in hash_to_cid:
                hash_to_cid[cate] = len(hash_to_cid)
    num_cates = len(hash_to_cid)

    item_cate = np.zeros((num_items, num_cates), dtype=np.bool)
    with open(os.path.join(data_dir, 'item_cate.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for item, cate in reader:
            item, cate = int(item), int(cate)
            if item not in hash_to_sid:
                continue
            item = hash_to_sid[item]
            cate = hash_to_cid[cate]
            item_cate[item, cate] = True
    item_cate = item_cate.astype(np.int64)

    js = np.argsort(item_cate.sum(axis=0))[-7:]
    item_cate = item_cate[:, js]
    assert np.min(np.sum(item_cate, axis=1)) == 1
    assert np.max(np.sum(item_cate, axis=1)) == 1
    return item_cate
