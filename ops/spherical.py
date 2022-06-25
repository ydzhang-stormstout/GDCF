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

from typing import Tuple, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .common import sqrt, e_i, expand_proj_dims
from .manifold import RadiusManifold
import numpy as np


class Sphere(RadiusManifold):
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
        return 1. / (self.radius**2)
    
    def cdist(self, x, y):
        # print(x.shape, y.shape)
        # xy_inner = torch.cdist(x, y)
        xy_inner = torch.matmul(x, y.t())
        # xy = torch.acos(xy_inner)
        # print(torch.max(self.curvature * xy_inner), torch.min(self.curvature * xy_inner))
        # assert torch.isfinite(xy).all() or torch.isnan(xy).all()
        # return torch.acos(xy_inner)
        return self.radius * torch.acos(self.curvature * xy_inner)


def _logdet(u: Tensor, radius: Tensor) -> Tensor:
    assert torch.isfinite(u).all()
    # det [(\partial / \partial v) proj_{\mu}(v)] = (R|sin(r)| / r)^(n-1)
    r = torch.norm(u, dim=-1, p=2) / radius
    n = u.shape[-1] - 1

    logdet_partial = (n - 1) * (torch.log(torch.abs(torch.sin(r)).clamp(min=1e-5)) -
                                torch.log(r.clamp(min=1e-5)))
#    logdet_partial = (n - 1) * (torch.log(radius) + torch.log(torch.abs(torch.sin(r)).clamp(min=1e-5)) -
#                                torch.log(r.clamp(min=1e-5)))
    assert torch.isfinite(logdet_partial).all()
    return logdet_partial


def mu_0(shape: torch.Size, radius: Tensor, **kwargs: Any) -> Tensor:
    return e_i(i=0, shape=shape, **kwargs) * radius


def parallel_transport_mu0(v: Tensor, dst: Tensor, radius: Tensor) -> Tensor:
    coef = torch.sum(dst * v, dim=-1, keepdim=True) / torch.clamp(input=radius * (radius + dst[..., 0:1]), min=1e-6)
    right = torch.cat((dst[..., 0:1] + radius, dst[..., 1:]), dim=-1)
    # print(coef)
    # if not torch.isfinite(coef).all():
    #    #_coef = coef.to('cpu').detach().numpy()
    #    #print(np.argwhere(np.isinf(_coef))) [170, 0]
    #    #print(torch.isfinite(coef))
    #    print(dst.shape, v.shape)
    #    _dst = dst[170]
    #    _v = v[170]
    #    print(torch.sum(_dst * _v, dim=-1))
    #    print(radius * (radius + dst[170, 0:1]))
    #    print(dst[170], v[170], radius)
    assert torch.isfinite(right).all()
    assert torch.isfinite(coef).all()
    return v - coef * right


def inverse_parallel_transport_mu0(x: Tensor, src: Tensor, radius: Tensor) -> Tensor:
    coef = x[..., 0:1] / (radius + src[..., 0:1])
    right = torch.cat((src[..., 0:1] + radius, src[..., 1:]), dim=-1)
    return x - coef * right


def exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    assert torch.isfinite(x).all()
    assert torch.isfinite(at_point).all()
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True) / radius
    x_normed = x / x_norm
    ret = torch.cos(x_norm) * at_point + torch.sin(x_norm) * x_normed
    assert torch.isfinite(ret).all()
    return ret


def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    assert x[..., 0].allclose(torch.zeros_like(x[..., 0]))
    x = x[..., 1:]
    x_norm = torch.norm(x, p=2, keepdim=True, dim=-1) / radius
    x_normed = F.normalize(x, p=2, dim=-1) * radius
    ret = torch.cat((torch.cos(x_norm) * radius, torch.sin(x_norm) * x_normed), dim=-1)
    assert torch.isfinite(ret).all()
    return ret


def inverse_exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    alpha = torch.sum(at_point * x, dim=-1, keepdim=True) / (radius**2)
    coef = torch.acos(torch.clamp(alpha, min=-1., max=1.)) / sqrt(1. - alpha**2)
    ret = coef * (x - alpha * at_point)
    assert torch.isfinite(ret).all()
    return ret


def inverse_exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    alpha = x[..., 0:1] / radius  # dot(x, mu0, keepdim=True) / R^2 .. <x, mu0> = x[0] * R
    coef = torch.acos(torch.clamp(alpha, min=-1., max=1.)) / sqrt(1. - alpha**2)
    diff = torch.cat((x[..., 0:1] - alpha * radius, x[..., 1:]), dim=-1)  # y - alpha*mu0 = (y[0]-alpha(-R); y[1:])
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


def spherical_to_projected(x: Tensor, radius: Tensor) -> Tensor:
    return radius * x[..., 1:] / (radius + x[..., 0:1])
