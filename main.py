import torch
from config import parser
import numpy as np
import logging
import datetime
import os
from Geo_vae import GeoVAE
from tensorboardX import SummaryWriter
import time
from utils.data_utils import load_data
from utils.metric import ndcg_binary_at_k_batch, recall_at_k_batch
from scipy import sparse
from torch.optim import Adam
from geoopt.optim import RiemannianAdam
import traceback

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

    (_, train_data, vad_data_tr, vad_data_te, tst_data_tr, tst_data_te) = load_data(args.dataset)

    logging.getLogger().setLevel(logging.INFO)

    total_dim = args.dim * args.k
    log_dir = 'log/{}/{}/{}'.format(args.dataset, args.manifold, args.dim)
    if 'unite' in args.manifold:
        log_dir = os.path.join(log_dir, args.component)
    log_dir = os.path.join(log_dir, 'lr{}_rg{}_e{}_b{}_dr{}_beta{}_tau{}_std{}_k{}_d{}_gb{}_seed{}'.format(
        args.lr, args.rg, args.epoch,
        args.batch, args.dropout, args.beta, args.tau, args.std, 
        args.k, args.dim, args.nogb, args.seed
    ))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.tensorboard:
        tb: SummaryWriter = SummaryWriter(
                                    os.path.join(log_dir, 'tensorboard_{}'.format(
                                    time.strftime("%m-%d-%Hh%Mm%Ss"))
                                    ))
    logging.basicConfig(level=logging.INFO, 
                       handlers=[logging.FileHandler(os.path.join(log_dir, 'log.txt')),
                               logging.StreamHandler()
                           ])
    logging.info('Using cuda: {}'.format(args.cuda))
    logging.info('Using seed: {}'.format(args.seed))
    logging.info('Using dataset: {}'.format(args.dataset))

    n = train_data.shape[0]
    n_items = train_data.shape[1]
    idxlist = list(range(n))
    batch_size = args.batch

    num_batches = int(np.ceil(float(n) / args.batch))
    total_anneal_steps = 5 * num_batches
    print(num_batches)

    n_vad = vad_data_tr.shape[0]
    idxlist_vad = list(range(n_vad))
    n_test = tst_data_tr.shape[0]
    idxlist_test = list(range(n_test))
    
    if args.manifold == 'unite':
        model = GeoVAE(args, n_items)
    elif args.manifold == 'universal':
        model = kappaVAE(args, n_items)
    else:
        raise NotImplementedError

    print('load model finished!')
    if int(args.cuda) >= 0:
        model = model.to(args.device)

    
    best_val_loss = np.inf
    best_val_ndcg = - np.inf
    ndcg20_test = - np.inf
    ndcg50_test = - np.inf
    ndcg100_test = - np.inf
    recall20_test = - np.inf
    recall50_test = - np.inf
    recall100_test = - np.inf
    ndcg20_test_std = - np.inf
    ndcg50_test_std = - np.inf
    ndcg100_test_std = - np.inf
    recall20_test_std = - np.inf
    recall50_test_std = - np.inf
    recall100_test_std = - np.inf

    log_manifold = args.manifold
    if 'unite' in args.manifold:
        log_manifold += '_' + args.component

    print('Trainable parameters in this model')
    model_parameters = []
    model_parameters.append({'params': model.parameters()})
    if args.manifold == 'universal':
        for _manifold in model.manifolds:
            model_parameters.append({'params': _manifold.parameters()})
    print(model_parameters)

    for name, para in model.named_parameters():
        if para.requires_grad:
            print('trainable', name)
        else:
            print('non-trainable', name)
    if args.manifold == 'universal':
        for _manifold in model.manifolds:
            for name, para in _manifold.named_parameters():
                if para.requires_grad:
                    print('trainable', name)
                else:
                    print('non-trainable', name)
  

    update_count = 0.
    optimizer = RiemannianAdam(params=model_parameters, lr=args.lr)


    try:
        for epoch in range(args.epoch):
            np.random.shuffle(idxlist)
            model.train()
            if args.manifold == 'universal':
                if (epoch + 0) % 10 == 0:
                    curvature = [round(_.k.to('cpu').detach().numpy().item(), 5) for _ in model.manifolds]
                    logging.info(curvature)
    
            aver_loss = 0.
            for bnum, st_idx in enumerate(range(0, n, batch_size)):
                end_idx = min(st_idx + batch_size, n)
                x = train_data[idxlist[st_idx: end_idx]]
                if sparse.isspmatrix(x):
                    x = x.toarray()
                x = x.astype('float32')
                x = torch.from_numpy(x).to(args.device)
                if total_anneal_steps > 0:
                    anneal = min(args.beta, 1. * update_count / total_anneal_steps)
                else:
                    anneal = args.beta
                model.input(x, is_train=1, anneal_ph=anneal)
                logits, loss = model.build_loss()
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                aver_loss += loss.cpu().item()
                update_count += 1
            aver_loss /= (bnum + 1)
    
            ndcg_val_dist = []
            model.eval()
            val_loss = 0.
            val_top_k = 20
            for bnum, st_idx in enumerate(range(0, n_vad, batch_size)):
                end_idx = min(st_idx + batch_size, n_vad)
                x = vad_data_tr[idxlist_vad[st_idx:end_idx]]
                if sparse.isspmatrix(x):
                    x = x.toarray()
                x_cpu = x.astype('float32')
                x = torch.from_numpy(x_cpu).to(args.device)
                model.input(x, is_train=0, anneal_ph=args.beta)
                logits, loss = model.build_loss()
                logits = logits.to('cpu').detach().numpy()
                logits[x_cpu.nonzero()] = -np.inf
                ndcg_val_dist.append(ndcg_binary_at_k_batch(logits, vad_data_te[idxlist_vad[st_idx:end_idx]], k=val_top_k))
                val_loss += loss.cpu().item()
            val_loss /= (bnum + 1)
            ndcg_val_dist = np.concatenate(ndcg_val_dist)
            ndcg_val = ndcg_val_dist.mean()
            if ndcg_val > best_val_ndcg or val_loss < best_val_loss:
                if ndcg_val > best_val_ndcg:
                    best_val_ndcg = ndcg_val
                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
    
                # Test
                ndcg20, ndcg50, ndcg100, recall20, recall50, recall100 = [], [], [], [], [], []
                test_loss = 0.
                for bnum, st_idx in enumerate(range(0, n_test, batch_size)):
                    end_idx = min(st_idx + batch_size, n_test)
                    x = tst_data_tr[idxlist_test[st_idx:end_idx]]
                    if sparse.isspmatrix(x):
                        x = x.toarray()
                    x_cpu = x.astype('float32')
                    x = torch.from_numpy(x_cpu).to(args.device)
                    model.input(x, is_train=0, anneal_ph=args.beta)
                    logits, loss = model.build_loss()
                    logits = logits.to('cpu').detach().numpy()
                    logits[x_cpu.nonzero()] = -np.inf
    
                    ndcg20.append(ndcg_binary_at_k_batch(
                        logits, tst_data_te[idxlist_test[st_idx:end_idx]], k=20))
                    ndcg50.append(ndcg_binary_at_k_batch(
                        logits, tst_data_te[idxlist_test[st_idx:end_idx]], k=50))
                    ndcg100.append(ndcg_binary_at_k_batch(
                        logits, tst_data_te[idxlist_test[st_idx:end_idx]], k=100))
                    recall20.append(recall_at_k_batch(
                        logits, tst_data_te[idxlist_test[st_idx:end_idx]], k=20))
                    recall50.append(recall_at_k_batch(
                        logits, tst_data_te[idxlist_test[st_idx:end_idx]], k=50))
                    recall100.append(recall_at_k_batch(
                        logits, tst_data_te[idxlist_test[st_idx:end_idx]], k=100))
    
                ndcg20 = np.concatenate(ndcg20)
                ndcg50 = np.concatenate(ndcg50)
                ndcg100 = np.concatenate(ndcg100)
                recall20 = np.concatenate(recall20)
                recall50 = np.concatenate(recall50)
                recall100 = np.concatenate(recall100)
    
                ndcg20_test, ndcg20_test_std = ndcg20.mean(), np.std(ndcg20) / np.sqrt(len(ndcg20))
                ndcg50_test, ndcg50_test_std = ndcg50.mean(), np.std(ndcg50) / np.sqrt(len(ndcg50))
                ndcg100_test, ndcg100_test_std = ndcg100.mean(), np.std(ndcg100) / np.sqrt(len(ndcg100))
                recall20_test, recall20_test_std = recall20.mean(), np.std(recall20) / np.sqrt(len(recall20))
                recall50_test, recall50_test_std = recall50.mean(), np.std(recall50) / np.sqrt(len(recall50))
                recall100_test, recall100_test_std = recall100.mean(), np.std(recall100) / np.sqrt(len(recall100))
    
                test_loss += loss.cpu().item()
                test_loss /= (bnum + 1)
    
            logging.info(" ".join(['epoch'+str(epoch), args.dataset, log_manifold, 
                                'train', f'loss {aver_loss:.3f}',
                                'val', f'loss {val_loss:.3f}', f'ndcg@{val_top_k}:{ndcg_val:.3f}']))
        logging.info(' '.join(['Validation', 'stop', 'NDCG@{}:{}'.format(val_top_k, ndcg_val),
                              'best', 'NDCG@{}:{}'.format(val_top_k, best_val_ndcg)]))
        logging.info(' '.join(['Test', 
                     'NDCG@20:%.5f+%.5f' % (ndcg20_test, ndcg20_test_std),
                     'NDCG@50:%.5f+%.5f' % (ndcg50_test, ndcg50_test_std),
                     'NDCG@100:%.5f+%.5f' % (ndcg100_test, ndcg100_test_std),
                     'Recall@20:%.5f+%.5f' % (recall20_test, recall20_test_std),
                     'Recall@50:%.5f+%.5f' % (recall50_test, recall50_test_std),
                     'Recall@100:%.5f+%.5f' % (recall100_test, recall100_test_std)
        ]))
    except Exception as e:
        logging.info('Running error')
        logging.info(traceback.format_exc())
    finally:
        pass
           
def compute_k(args):
    if not 'unite' in args.manifold:
        return args.k
    k = 0
    for _component in args.component.split(','):
        k += int(_component[1:])
    return k


if __name__ == '__main__':
    args = parser.parse_args()
    args.k = compute_k(args)
    train(args)
