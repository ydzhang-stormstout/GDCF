import numpy as np
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch import optim
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import geoopt
from geoopt import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import Sphere as geo_Sphere, Lorentz as geo_Hyperboloid, SphereProjection as geo_ShpereProj, PoincareBall as geo_PoincareBall, Euclidean as geo_Euclidean
from ops import Sphere as my_Sphere, Hyperboloid as my_Hyperboloid, StereographicallyProjectedSphere as my_ShpereProj, PoincareBall as my_PoincareBall, Euclidean as my_Euclidean


class GeoVAE(nn.Module):
    def __init__(self, args, num_items):
        super().__init__()
        self.rg = args.rg
        self.random_seed = args.seed
        self.num_items = num_items
        self.batch_size = args.batch
        self.std0 = args.std
        self.tau = args.tau
        self.tau_list = []
        self.nogb = args.nogb
        self.k = 0
        self.dim = args.dim
        self.dropout = float(args.dropout)

        self.q_dims = [num_items, self.dim, self.dim]
        weights_q, biases_q = [], []
        self.weights_q = nn.ParameterList()
        self.biases_q = nn.ParameterList()
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2
            self.weights_q.append(Parameter(Tensor(d_in, d_out)))
            nn.init.xavier_uniform_(self.weights_q[-1], gain=1.)
            self.biases_q.append(Parameter(Tensor(d_out)))
            nn.init.normal_(self.biases_q[-1])

        self.items = Parameter(Tensor(num_items, self.dim))
        nn.init.xavier_uniform_(self.items)

        self.cores = nn.ParameterList()
        self.my_manifolds = []
        self.geo_manifolds = []
        tmp_sphere = geo_manifold_dict['s']()
        for _component in args.component.split(','):
            s = _component[0:1]
            _num = int(_component[1:])
            for _ in range(_num):
                self.k += 1
                _radius = nn.Parameter(torch.tensor(1.), requires_grad=False)
                tmp_manifold = geo_manifold_dict[s[0]]() if s[0] != 's' else geo_manifold_dict['e']()
                _manifold = my_manifold_dict[s[0]](lambda: _radius) if s[0]!='e' else my_manifold_dict[s[0]]()
                self.geo_manifolds.append(tmp_manifold)
                self.my_manifolds.append(_manifold)
                _dim = self.dim if _manifold.is_project_manifold() else self.dim + 1
                _tensor = Tensor(1, _dim)
                nn.init.xavier_uniform_(_tensor)
                _tensor = tmp_manifold.projx(_tensor) if s[0] != 's' else tmp_sphere.projx(_tensor)
                if s[0] == 's':
                    _tensor[:,0] = torch.abs(_tensor[:,0])
                self.cores.append(ManifoldParameter(_tensor, manifold=tmp_manifold))
                self.tau_list.append(self.tau * tau_radio_dict[s[0]])
    
    def input(self, x, is_train, anneal_ph):
        self.input_ph = x
        self.input_ph.requires_grad = False
        self.is_training_ph = is_train
        if self.is_training_ph:
            self.anneal_ph = anneal_ph
    
    def build_loss(self, save_emb=False):
        if save_emb:
            facets_list = self.forward_pass(save_emb=True)
            return facets_list, self.items, self.cores
        logits, recon_loss, kl = self.forward_pass(save_emb=False)
        reg_var = torch.norm(self.items)
        for _ in self.weights_q:
            reg_var = reg_var + torch.norm(_)
        for _ in self.cores:
            reg_var = reg_var + torch.norm(_)

        neg_elbo = recon_loss + self.anneal_ph * kl + self.rg * reg_var
        return logits, neg_elbo

    def q_graph_k(self, x: Tensor):
        h = F.normalize(x, p=2, dim=1)
        h = F.dropout(h, self.dropout, training=self.training)
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = torch.matmul(h, w) + b
            if i != len(self.weights_q) - 1:
                h = torch.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = self.std0
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
                kl = torch.mean(torch.sum(0.5 * (-lnvarq_sub_lnvar0 + torch.exp(lnvarq_sub_lnvar0) - 1. ), dim=1))
        return mu_q, std_q, kl

    def forward_pass(self, save_emb):
        cates_logits = None
        for k, core_k in enumerate(self.cores):
            _my_manifold = self.my_manifolds[k]
            _geo_manifold = self.geo_manifolds[k]
            _items = self.items
            assert torch.isfinite(_items).all() or torch.isnan(_items).all()
            if not _my_manifold.is_project_manifold():
                _zeros = torch.zeros(self.items.shape[0], 1, dtype=self.items.dtype, device=self.items.device)
                _items = torch.cat((_zeros, _items), dim=1)
            _items = _my_manifold.parallel_transport_mu0(x=_items, dst=core_k)
            try:
                _cates = _geo_manifold.norm(u=_items, keepdim=True)
            except:
                _cates = _geo_manifold.norm(u=_items, x=core_k, keepdim=True)
            _cates = _cates / self.tau_list[k]
            cates_logits = _cates  if cates_logits == None else torch.cat((cates_logits, _cates), dim=1)
            assert torch.isfinite(cates_logits).all()
        cates_logits = - cates_logits

        assert torch.isfinite(cates_logits).all()
        if self.nogb:
            cates = torch.softmax(cates_logits, dim=1)
        else:
            cates_dist = RelaxedOneHotCategorical(1, logits=cates_logits) # logists? pro?
            cates_sample = cates_dist.sample()
            cates_mode = torch.softmax(cates_logits, dim=1)
            cates = (self.is_training_ph * cates_sample + (1 - self.is_training_ph) * cates_mode)
        
        z_list = []
        probs, kl = None, None
        for k, core_k in enumerate(self.cores):
            _my_manifold = self.my_manifolds[k]
            _geo_manifold = self.geo_manifolds[k]
            assert torch.isfinite(cates).all()
            cates_k = torch.reshape(cates[:, k], (1, -1))
            # q-net
            x_k = self.input_ph * cates_k
            mu_k, std_k, kl_k = self.q_graph_k(x_k)
            mu_k = _my_manifold.exp_map_mu0(mu_k)
            epsilon = torch.randn_like(std_k)
            z_k = mu_k
            if self.is_training_ph:
                _std_k = epsilon * std_k
                if not _my_manifold.is_project_manifold():
                    _zeros = torch.zeros(_std_k.shape[0], 1, dtype=_std_k.dtype, device=_std_k.device)
                    _std_k = torch.cat((_zeros, _std_k), dim=1)
                _std_k = _my_manifold.parallel_transport_mu0(x=_std_k, dst=mu_k)
                z_k = _my_manifold.exp_map(x=_std_k, at_point=mu_k)

            kl = (kl_k if (kl is None) else (kl + kl_k))
            if save_emb:
                z_list.append(z_k)
            # p-net
            _items = self.items
            if not _my_manifold.is_project_manifold():
                _zeros = torch.zeros(_items.shape[0], 1, dtype=_items.dtype, device=_items.device)
                _items = torch.cat((_zeros, _items), dim=1)

            v_k = _my_manifold.inverse_exp_map_mu0(x=z_k)
            v_k_unite = F.normalize(v_k, p=2, dim=1)
            items_unite = F.normalize(_items, p=2, dim=1)
            logits_k = torch.matmul(v_k_unite, items_unite.t()) / self.tau

            probs_k = torch.exp(logits_k.clamp(max=50))
            assert torch.isfinite(probs_k).all() or torch.isnan(probs_k).all()
            probs_k = probs_k * cates_k
            probs = (probs_k if (probs is None) else (probs + probs_k))
            assert torch.isfinite(probs).all() or torch.isnan(probs).all()
        logits = torch.log(probs)
        logits = torch.log_softmax(logits, -1)
        assert torch.isfinite(logits).all() or torch.isnan(logits).all()
        recon_loss = torch.mean(torch.sum(-logits * self.input_ph, dim=-1))
        if save_emb:
            return z_list

        return logits, recon_loss, kl

tau_radio_dict = {
    'e': 1,
    'h': 1,
    's': 0.2,
    'p': 1,
    'd': 0.3
}
my_manifold_dict = {
    'e': my_Euclidean,
    'h': my_Hyperboloid,
    's': my_Sphere,
    'd': my_ShpereProj,
    'p': my_PoincareBall
}
geo_manifold_dict = {
    'e': geo_Euclidean,
    'h': geo_Hyperboloid,
    's': geo_Sphere,
    'd': geo_ShpereProj,
    'p': geo_PoincareBall
}


if __name__ == '__main__':
    a = GeoVAE()
