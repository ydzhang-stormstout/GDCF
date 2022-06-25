# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Any, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .common import acosh, cosh, sinh, sqrt, logsinh, e_i, expand_proj_dims
from .manifold import RadiusManifold


class Hyperboloid(RadiusManifold):
    def is_project_manifold(self):
        return False

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        return exp_map_mu0(expand_proj_dims(x), radius=self.radius)
    
    def exp_map(self, x: Tensor, at_point: Tensor) -> Tensor:
        return exp_map(x, at_point, radius=self.radius)

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        return inverse_exp_map_mu0(x, radius=self.radius)
    
    def inverse_exp_map(self, x: Tensor, at_point: Tensor) -> Tensor:
        return inverse_exp_map(x=x, at_point=at_point, radius=self.radius)

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        return parallel_transport_mu0(x, dst, radius=self.radius)

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        return inverse_parallel_transport_mu0(x, src, radius=self.radius)

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        return mu_0(shape, radius=self.radius, **kwargs)

    def sample_projection_mu0(self, x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return sample_projection_mu0(x, at_point, radius=self.radius)

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        return inverse_sample_projection_mu0(x_proj, at_point, radius=self.radius)

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        u = data[0]
        return _logdet(u, self.radius)

    @property
    def curvature(self) -> Tensor:
        return -super().curvature
    
    def cdist(self, x, y):
        c = - self.curvature
        assert torch.isfinite(y).all() or torch.isnan(y).all()
        assert torch.isfinite(x).all() or torch.isnan(x).all()
        xy_inner = _l_inner(x, y)
        # print(xy_inner)
        assert torch.isfinite(xy_inner).all() or torch.isnan(xy_inner).all()
        res = self.radius * torch.acosh(torch.clamp_min(c * xy_inner, 1 + 1e-8))
        assert torch.isfinite(res).all() or torch.isnan(res).all()
        return res



def _l_inner(x, y):
    # x: [n, d], y: [m, d]
    #print('shape11', x.shape, y.shape)
    res_matrix = torch.mm(x, y.t())
    res_matrix -= 2 * torch.mm(x[:,:1], y[:,:1].t())
    #print('shape', res_matrix.shape)
    return res_matrix 


def _logdet(u: Tensor, radius: Tensor) -> Tensor:
    # det [(\partial / \partial v) proj_{\mu}(v)] = (Rsinh(r) / r)^(n-1)
    r = lorentz_norm(u, dim=-1) / radius
    n = u.shape[-1] - 1

    logdet_partial = (n - 1) * (logsinh(r) - torch.log(r))
    # original error logdet_partial = (n - 1) * (torch.log(radius) + logsinh(r) - torch.log(r))
    assert torch.isfinite(logdet_partial).all()
    return logdet_partial


max_norm = 1000


def _radis2c(r):
    return r ** 2


def _normalize(p, c):
    """
    Normalize vector to confirm it is located on the hyperboloid
    :param p: [nodes, features(d + 1)]
    :param c: parameter about curvature
    """
    d = p.size(-1) - 1
    narrowed = p.narrow(-1, 1, d)
    narrowed = torch.renorm(narrowed.view(-1, d), 2, 0, max_norm)
    first = c + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
    first = torch.sqrt(first)
    return torch.cat((first, narrowed), dim=1)

def _normalize_tangent(p, p_tan, c):
    """
    Normalize tangent vectors to place the vectors satisfies <p, p_tan>_L=0
    :param p: the tangent spaces at p. size:[nodes, feature]
    :param p_tan: the tangent vector in tangent space at p
    """
    d = p_tan.size(1) - 1
    p_tail = p.narrow(1, 1, d)
    p_tan_tail = p_tan.narrow(1, 1, d)
    ptpt = torch.sum(p_tail * p_tan_tail, dim=1, keepdim=True)
    p_head = torch.sqrt(c + torch.sum(torch.pow(p_tail, 2), dim=1, keepdim=True))
    return torch.cat((ptpt / p_head, p_tan_tail), dim=1)

def _normalize_tangent_zero(p_tan, c):
    zeros = torch.zeros_like(p_tan)
    zeros[:, 0] = c ** 0.5
    return normalize_tangent(zeros, p_tan, c)



def mu_0(shape: Tuple[int, ...], radius: Tensor, **kwargs: Any) -> Tensor:
    return e_i(i=0, shape=shape, **kwargs) * radius


def lorentz_product(x: Tensor, y: Tensor, keepdim: bool = False, dim: int = -1) -> Tensor:
    m = x * y
    if keepdim:
        ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
    else:
        ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
    return ret


def lorentz_norm(x: Tensor, **kwargs: Any) -> Tensor:
    product = lorentz_product(x, x, **kwargs)
    ret = sqrt(product)
    return ret


def parallel_transport_mu0(x: Tensor, dst: Tensor, radius: Tensor) -> Tensor:
    # PT_{mu0 -> dst}(x) = x + <dst, x>_L / (R^2 - <mu0, dst>_L) * (mu0+dst)
    denom = radius * (radius + dst[..., 0:1])  # lorentz_product(mu0, dst, keepdim=True) which is -dst[0]*radius
    lp = lorentz_product(dst, x, keepdim=True)
    coef = lp / denom
    right = torch.cat((dst[..., 0:1] + radius, dst[..., 1:]), dim=-1)  # mu0 + dst
    return x + coef * right


def inverse_parallel_transport_mu0(x: Tensor, src: Tensor, radius: Tensor) -> Tensor:
    # PT_{src -> mu0}(x) = x + <mu0, x>_L / (R^2 - <src, mu0>_L) * (src+mu0)
    denom = (radius + src[..., 0:1])  # lorentz_product(src, mu0, keepdim=True) which is -src[0]*radius
    lp = -x[..., 0:1]  # lorentz_product(mu0, x, keepdim=True) which is -x[0]*radius
    # coef = (lp * radius) / (radius * denom)
    coef = lp / denom
    right = torch.cat((src[..., 0:1] + radius, src[..., 1:]), dim=-1)  # mu0 + src
    return x + coef * right


def exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    x_norm = lorentz_norm(x, keepdim=True) / radius
    x_normed = x / x_norm
    ret = cosh(x_norm) * at_point + sinh(x_norm) * x_normed
    return _normalize(ret, _radis2c(radius))
    # assert torch.isfinite(ret).all()
    # return ret


def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    assert x[..., 0].allclose(torch.zeros_like(x[..., 0]))
    x = x[..., 1:]
    x_norm = torch.norm(x, p=2, keepdim=True, dim=-1) / radius
    x_normed = F.normalize(x, p=2, dim=-1) * radius
    ret = torch.cat((cosh(x_norm) * radius, sinh(x_norm) * x_normed), dim=-1)
    # assert torch.isfinite(ret).all()
    return _normalize(ret, _radis2c(radius))
    # return ret


def inverse_exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    alpha = -lorentz_product(at_point, x, keepdim=True) / (radius**2)
    coef = acosh(alpha) / sqrt(alpha**2 - 1)
    ret = coef * (x - alpha * at_point)
    return ret


def inverse_exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    alpha = x[..., 0:1] / radius  # -lorentz_product(x, mu0, keepdim=True) / R^2 .. -<x, mu0>_L = x[0] * R
    coef = acosh(alpha) / sqrt(alpha**2 - 1.)
    diff = torch.cat((x[..., 0:1] - alpha * radius, x[..., 1:]), dim=-1)
    return coef * diff


def sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x_expanded = expand_proj_dims(x)
    pt = parallel_transport_mu0(x_expanded, dst=at_point, radius=radius)
    x_proj = exp_map(pt, at_point=at_point, radius=radius)
    return x_proj, (pt, x)


def inverse_sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tensor]:
    unmapped = inverse_exp_map(x, at_point=at_point, radius=radius)
    unpt = inverse_parallel_transport_mu0(unmapped, src=at_point, radius=radius)
    return unmapped, unpt[..., 1:]


def lorentz_to_poincare(x: Tensor, radius: Tensor) -> Tensor:
    return radius * x[..., 1:] / (radius + x[..., 0:1])
