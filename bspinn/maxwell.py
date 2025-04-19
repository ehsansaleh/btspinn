# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## The Maxwell-Ampere Problem Script

# %% tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import importlib
# %matplotlib inline
# if importlib.util.find_spec("matplotlib_inline") is not None:
#     import matplotlib_inline
#     matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
# else:
#     from IPython.display import set_matplotlib_formats
#     set_matplotlib_formats('retina')
#
# plt.ioff();

# %% tags=["active-ipynb"]
# # This cell is a sorry attempt at integrating the code onto google colab.
# # Feel free to remove this cell if you are not using google colab.
#
# import os, sys, sysconfig
# from importlib.util import find_spec
# is_colab = 'google.colab' in sys.modules
# site_userpkg = sysconfig.get_paths()['purelib']
# req_pkgscolab = ['tensorboardX', 'pyinstrument', 'chaospy']
# if is_colab and any(find_spec(pkg) is None for pkg in req_pkgscolab):
#     ! pip install {" ".join(req_pkgscolab)}
# if is_colab and not os.path.exists('/content/btspinn/'):
#     ! cd /content && git clone https://github.com/ehsansaleh/btspinn.git
#     ! cd /content/btspinn && pip install -e . && rm -rf *.egg-info
# if is_colab and (find_spec('bspinn') is None):
#     ! ln -s /content/btspinn/bspinn {site_userpkg}/bspinn
# elif is_colab:
#     ! [ -L {site_userpkg}/bspinn ] && rm {site_userpkg}/bspinn
# if is_colab and os.path.exists('/content/btspinn/notebook'):
#     os.chdir('/content/btspinn/notebook')

# %%
import numpy as np
import torch
import os
import json
import yaml
import time
import shutil
import socket
import random
import pathlib
import fnmatch
import datetime
import resource
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import psutil
from pyinstrument import Profiler
from torch import nn
from copy import deepcopy
from itertools import chain
from scipy.special import gamma
from os.path import exists, isdir
from itertools import product
from collections import defaultdict
from collections import OrderedDict as odict
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
from bspinn.io_utils import DataWriter
from bspinn.io_utils import get_git_commit
from bspinn.io_utils import preproc_cfgdict
from bspinn.io_utils import hie2deep, deep2hie

from bspinn.tch_utils import isscalar
from bspinn.tch_utils import EMA
from bspinn.tch_utils import BatchRNG
from bspinn.tch_utils import bffnn
from bspinn.tch_utils import profmem

from bspinn.io_cfg import configs_dir
from bspinn.io_cfg import results_dir
from bspinn.io_cfg import storage_dir

# Disabling pytorch deprecation warnings
import warnings
warnings.filterwarnings("ignore")


# %% [markdown]
# ## Theory
#
# Consider the $3$-dimensional space $\mathbb{R}^{3}$, and the following current along the z-axis:
#
# $$\vec{J}(x) = I \cdot \delta^2(x_1=0, x_2=0, x_3\in [z_1, z_2]).$$
#
# The analytical solution to the system
#
# $$\nabla \times \vec{A} = \vec{B}$$
#
# $$\nabla \times \vec{B} = \vec{J}$$
#
# can be expressed as
#
# $$\vec{A} = \frac{-I}{4\pi} \begin{bmatrix}0\\0\\ 
# \log\bigg(\frac{(z_2-x_3) + 
# \sqrt{x_1^2 + x_2^2 + (z_2-x_3)^2}}{(z_1-x_3) + 
# \sqrt{x_1^2 + x_2^2 + (z_1 - x_3)^2}}\bigg)\end{bmatrix},$$
#
# and
#
# $$\vec{B} = \frac{-I}{4\pi\cdot \sqrt{x_1^2+x_2^2}} \cdot 
# \bigg(\frac{z_2-x_3}{\sqrt{x_1^2 + x_2^2 + (z_2-x_3)^2}} - 
# \frac{z_1-x_3}{\sqrt{x_1^2 + x_2^2 + (z_1-x_3)^2}}\bigg) \cdot 
# \begin{bmatrix}\frac{-x_2}{\sqrt{x_1^2 + x_2^2}}\\
# \frac{x_1}{\sqrt{x_1^2 + x_2^2}}\\0\end{bmatrix}.$$
#
#
#

# %% [markdown]
# ### Defining the Problem and the Analytical Solution

# %% code_folding=[1, 17, 80, 138]
class DeltaLineProblem:
    def __init__(self, source, sink, current, tch_device, tch_dtype):
        n_seeds, n_wires, dim = source.shape
        assert source.shape   == (n_seeds, n_wires, dim)
        assert sink.shape     == (n_seeds, n_wires, dim)
        assert current.shape  == (n_seeds, n_wires)
        assert dim == 3
        
        self.n_seeds = n_seeds
        self.n_wires = n_wires
        self.source = source
        self.sink = sink
        self.current = current
        self.source_tch = torch.from_numpy(source).to(tch_device, tch_dtype)
        self.sink_tch = torch.from_numpy(sink).to(tch_device, tch_dtype)
        self.current_tch = torch.from_numpy(current).to(tch_device, tch_dtype)
        
        self.shape = (self.n_seeds,)
        self.ndim = 1
        
    def integrate_volumes(self, volumes):
        n_seeds, n_wires = self.n_seeds, self.n_wires
        source, sink, current = self.source_tch, self.sink_tch, self.current_tch
        dim = 3
        
        assert volumes['type'] == 'disk'
        centers = volumes['centers']
        radii = volumes['radii']
        normals = volumes['normals'].squeeze(-1)
        
        n_vol = radii.shape[-1]
        
        assert centers.shape == (n_seeds, n_vol, dim), (centers.shape, (n_seeds, n_vol, dim))
        assert radii.shape   == (n_seeds, n_vol)
        assert normals.shape == (n_seeds, n_vol, dim)
        
        assert source.shape  == (n_seeds, n_wires, dim)
        assert sink.shape    == (n_seeds, n_wires, dim)
        assert current.shape == (n_seeds, n_wires)
        
        sink2cntr = centers.reshape(n_seeds, n_vol, 1, dim) - sink.reshape(n_seeds, 1, n_wires, dim)
        assert sink2cntr.shape == (n_seeds, n_vol, n_wires, dim)
        
        src2snk = source - sink
        assert src2snk.shape == (n_seeds, n_wires, dim)
        
        lam_numer = (sink2cntr * normals.reshape(n_seeds, n_vol, 1, dim)).sum(dim=-1)
        assert lam_numer.shape == (n_seeds, n_vol, n_wires)
        
        lam_denom = (src2snk.reshape(n_seeds, 1, n_wires, dim) * 
                     normals.reshape(n_seeds, n_vol,   1, dim)).sum(dim=-1)
        assert lam_denom.shape == (n_seeds, n_vol, n_wires)
        
        # The interpolation coefficient between source and sink that 
        # lies on the disk plane: $lambda * source  + (1 - lambda) sink$
        lam = lam_numer / lam_denom
        assert lam.shape == (n_seeds, n_vol, n_wires)
        
        lam_ss = lam.reshape(n_seeds, n_vol, n_wires, 1) * src2snk.reshape(n_seeds, 1, n_wires, dim)
        assert lam_ss.shape == (n_seeds, n_vol, n_wires, dim)
        
        # The difference between the wire projection and the volum center
        p2c = sink2cntr - lam_ss
        assert p2c.shape == (n_seeds, n_vol, n_wires, dim)
        
        # The projection to center radius
        p2c_r = p2c.square().sum(dim=-1).sqrt() 
        assert p2c_r.shape == (n_seeds, n_vol, n_wires)
        
        is_within_r = p2c_r < radii.reshape(n_seeds, n_vol, 1)
        assert is_within_r.shape == (n_seeds, n_vol, n_wires)
        
        is_within_ss = (lam <= 1) * (lam >= 0)
        assert is_within_ss.shape == (n_seeds, n_vol, n_wires)
        
        valid_cross = is_within_r * is_within_ss
        assert valid_cross.shape == (n_seeds, n_vol, n_wires)
        
        sign = (normals.reshape(n_seeds, n_vol, 1, dim) * 
                src2snk.reshape(n_seeds, 1, n_wires, dim)).sum(dim=-1).sign()
        assert sign.shape == (n_seeds, n_vol, n_wires)
        
        current_enc = (valid_cross * sign * current.reshape(n_seeds, 1, n_wires)).sum(dim=-1)
        assert current_enc.shape == (n_seeds, n_vol)
        
        return current_enc
    
    def potential(self, x, eps=1e-10):
        n_seeds, n_wires = self.n_seeds, self.n_wires
        source, sink, current = self.source_tch, self.sink_tch, self.current_tch
        
        n_seeds, n_x, dim = x.shape 
        assert x.shape == (n_seeds, n_x, dim)
        
        src2snk = sink - source
        assert src2snk.shape == (n_seeds, n_wires, dim)
        
        src2snk_dist = src2snk.norm(dim=-1, keepdim=True)
        assert src2snk_dist.shape == (n_seeds, n_wires, 1)
        
        src2snk_norm = src2snk / src2snk_dist
        assert src2snk_norm.shape == (n_seeds, n_wires, dim)
        
        x2src = source.reshape(n_seeds, 1, n_wires, dim) - x.reshape(n_seeds, n_x, 1, dim)
        assert x2src.shape == (n_seeds, n_x, n_wires, dim)
        
        z1 = (x2src * src2snk_norm.reshape(n_seeds, 1, n_wires, dim)).sum(dim=-1)
        assert z1.shape == (n_seeds, n_x, n_wires)
        
        x2snk = sink.reshape(n_seeds, 1, n_wires, dim) - x.reshape(n_seeds, n_x, 1, dim)
        assert x2snk.shape == (n_seeds, n_x, n_wires, dim)
        
        z2 = (x2snk * src2snk_norm.reshape(n_seeds, 1, n_wires, dim)).sum(dim=-1)
        assert z2.shape == (n_seeds, n_x, n_wires)
        
        numer = (z2 + x2snk.norm(dim=-1))
        assert numer.shape == (n_seeds, n_x, n_wires)
        
        numer = torch.clip(numer, eps, None)
        assert numer.shape == (n_seeds, n_x, n_wires)
        
        assert (numer > 0).all(), numer[numer <= 0]
        
        log_numer = numer.log()
        assert log_numer.shape == (n_seeds, n_x, n_wires)
        
        denom = (z1 + x2src.norm(dim=-1))
        assert denom.shape == (n_seeds, n_x, n_wires)
        
        denom = torch.clip(denom, eps, None)
        assert denom.shape == (n_seeds, n_x, n_wires)
        
        assert (denom > 0).all(), denom[denom <= 0]
        
        log_denom = denom.log()
        assert log_denom.shape == (n_seeds, n_x, n_wires)
        
        cst = src2snk_dist.reshape(n_seeds, 1, n_wires).log() * 2.0
        assert cst.shape == (n_seeds, 1, n_wires)
        
        A_z = (log_numer - log_denom - cst) * current.reshape(n_seeds, 1, n_wires) / (-4 * np.pi)
        assert A_z.shape == (n_seeds, n_x, n_wires)
        
        A_wires = A_z.reshape(n_seeds, n_x, n_wires, 1) * src2snk_norm.reshape(n_seeds, 1, n_wires, dim)
        assert A_wires.shape == (n_seeds, n_x, n_wires, dim)
        
        A = A_wires.sum(dim=-2)
        assert A.shape == (n_seeds, n_x, dim)
        
        return A
    
    def field(self, x):
        n_seeds, n_wires = self.n_seeds, self.n_wires
        source, sink, current = self.source_tch, self.sink_tch, self.current_tch
        
        n_seeds, n_x, dim = x.shape 
        assert x.shape == (n_seeds, n_x, dim)
        
        src2snk = sink - source
        assert src2snk.shape == (n_seeds, n_wires, dim)
        
        src2snk_dist = src2snk.norm(dim=-1, keepdim=True)
        assert src2snk_dist.shape == (n_seeds, n_wires, 1)
        
        src2snk_norm = src2snk / src2snk_dist
        assert src2snk_norm.shape == (n_seeds, n_wires, dim)
        
        x2src = source.reshape(n_seeds, 1, n_wires, dim) - x.reshape(n_seeds, n_x, 1, dim)
        assert x2src.shape == (n_seeds, n_x, n_wires, dim)
        
        z1 = (x2src * src2snk_norm.reshape(n_seeds, 1, n_wires, dim)).sum(dim=-1)
        assert z1.shape == (n_seeds, n_x, n_wires)
        
        x2snk = sink.reshape(n_seeds, 1, n_wires, dim) - x.reshape(n_seeds, n_x, 1, dim)
        assert x2snk.shape == (n_seeds, n_x, n_wires, dim)
        
        z2 = (x2snk * src2snk_norm.reshape(n_seeds, 1, n_wires, dim)).sum(dim=-1)
        assert z2.shape == (n_seeds, n_x, n_wires)
        
        x2snk_dist = x2snk.norm(dim=-1)
        assert x2snk_dist.shape == (n_seeds, n_x, n_wires)
        
        x2src_dist = x2src.norm(dim=-1)
        assert x2src_dist.shape == (n_seeds, n_x, n_wires)
        
        x12 = x2src - z1.reshape(n_seeds, n_x, n_wires, 1) * src2snk_norm.reshape(n_seeds, 1, n_wires, dim)
        assert x12.shape == (n_seeds, n_x, n_wires, dim)
        
        c1 = (z2 / x2snk_dist - z1 / x2src_dist) / x12.norm(dim=-1)
        assert c1.shape == (n_seeds, n_x, n_wires)
        
        c2 = c1 * current.reshape(n_seeds, 1, n_wires) / (-4 * np.pi)
        assert c2.shape == (n_seeds, n_x, n_wires)
        
        xss_cross = torch.cross(x2src, src2snk.reshape(n_seeds, 1, n_wires, dim), dim=-1)
        assert xss_cross.shape == (n_seeds, n_x, n_wires, dim)
        
        xss_cross_nrom = xss_cross / xss_cross.norm(dim=-1, keepdim=True)
        assert xss_cross_nrom.shape == (n_seeds, n_x, n_wires, dim)
        
        field_wires = xss_cross_nrom * c2.reshape(n_seeds, n_x, n_wires, 1)
        assert field_wires.shape == (n_seeds, n_x, n_wires, dim)
        
        field = field_wires.sum(dim=-2)
        assert field.shape == (n_seeds, n_x, dim)
        
        return field
    


# %% [markdown]
# ### Defining the Volume Sampler

# %% code_folding=[1, 150]
class DiskSampler:
    def __init__(self, c_dstr, c_params, r_dstr, r_params, n_dstr, n_params, batch_rng):
        assert isinstance(c_params, dict)
        for name, param in c_params.items():
            msg_ = f'center param {name} is not np.array'
            assert isinstance(param, np.ndarray), msg_
        
        assert isinstance(r_params, dict)
        for name, param in r_params.items():
            msg_ = f'radius param {name} is not np.array'
            assert isinstance(param, np.ndarray), msg_
            
        assert isinstance(n_params, dict)
        for name, param in n_params.items():
            msg_ = f'radius param {name} is not np.array'
            assert isinstance(param, np.ndarray), msg_

        self.batch_rng = batch_rng
        self.lib = batch_rng.lib
        
        ##############################################################
        ################# Center Sampling Parameters #################
        ##############################################################
        c_params_ = c_params.copy()
        self.c_dstr = c_dstr
        if c_dstr == 'uniform':
            c_low = c_params_.pop('low')
            c_high = c_params_.pop('high')
            
            n_bch, dim = c_low.shape
            
            self.c_low_np = c_low.reshape(n_bch, 1, dim)
            self.c_high_np = c_high.reshape(n_bch, 1, dim)
            self.c_size_np = (self.c_high_np - self.c_low_np)

            if self.lib == 'torch':
                self.c_low_tch = torch.from_numpy(self.c_low_np).to(
                    device=self.batch_rng.device, dtype=self.batch_rng.dtype)
                self.c_high_tch = torch.from_numpy(self.c_high_np).to(
                    device=self.batch_rng.device, dtype=self.batch_rng.dtype)
                self.c_size_tch = torch.from_numpy(self.c_size_np).to(
                    device=self.batch_rng.device, dtype=self.batch_rng.dtype)
            
            self.c_low = self.c_low_np if self.lib == 'numpy' else self.c_low_tch
            self.c_size = self.c_size_np if self.lib == 'numpy' else self.c_size_tch
        elif c_dstr == 'normal':
            c_loc = c_params_.pop('loc')
            c_scale = c_params_.pop('scale')
            
            n_bch, dim = c_loc.shape
            self.c_loc_np = c_loc.reshape(n_bch, 1, dim)
            self.c_scale_np = c_scale.reshape(n_bch, 1, 1)
            
            if self.lib == 'torch':
                self.c_loc_tch = torch.from_numpy(self.c_loc_np).to(
                    device=self.batch_rng.device, dtype=self.batch_rng.dtype)
                self.c_scale_tch = torch.from_numpy(self.c_scale_np).to(
                    device=self.batch_rng.device, dtype=self.batch_rng.dtype)
                
            self.c_loc = self.c_loc_np if self.lib == 'numpy' else self.c_loc_tch
            self.c_scale = self.c_scale_np if self.lib == 'numpy' else self.c_scale_tch
        elif c_dstr == 'ball':
            c_cntr = c_params_.pop('c')
            c_radi = c_params_.pop('r')
            
            n_bch, dim = c_cntr.shape
            self.c_cntr_np = c_cntr.reshape(n_bch, 1, dim)
            self.c_radi_np = c_radi.reshape(n_bch, 1, 1)
            
            if self.lib == 'torch':
                self.c_cntr_tch = torch.from_numpy(self.c_cntr_np).to(
                    device=self.batch_rng.device, dtype=self.batch_rng.dtype)
                self.c_radi_tch = torch.from_numpy(self.c_radi_np).to(
                    device=self.batch_rng.device, dtype=self.batch_rng.dtype)
                
            self.c_cntr = self.c_cntr_np if self.lib == 'numpy' else self.c_cntr_tch
            self.c_radi = self.c_radi_np if self.lib == 'numpy' else self.c_radi_tch
        else:
            raise ValueError(f'c_dstr="{c_dstr}" not implemented')
        
        msg_ = f'Some center parameters were left unused: {list(c_params_.keys())}'
        assert len(c_params_) == 0, msg_
            
        self.n_bch, self.d = n_bch, dim
        
        ##############################################################
        ################# Radius Sampling Parameters #################
        ##############################################################
        r_params_ = r_params.copy()
        r_low = r_params_.pop('low')
        r_high = r_params_.pop('high')
        
        if r_dstr == 'uniform':
            self.r_upow = 1.0
        elif r_dstr == 'unifdpow':
            self.r_upow = 1.0 / self.d
        else:
            raise ValueError(f'r_dstr={r_dstr} not implemented')

        r_low_rshp = r_low.reshape(self.n_bch, 1)
        r_high_rshp = r_high.reshape(self.n_bch, 1)
        assert (r_low >= 0.0).all()
        assert (r_high >= r_low).all()
        
        self.r_dstr = r_dstr
        self.r_low_np = np.power(r_low_rshp, 1.0/self.r_upow)
        self.r_high_np = np.power(r_high_rshp, 1.0/self.r_upow)
        self.r_size_np = (self.r_high_np - self.r_low_np)
        
        if self.lib == 'torch':
            self.r_low_tch = torch.from_numpy(self.r_low_np).to(
                device=self.batch_rng.device, dtype=self.batch_rng.dtype)
            self.r_high_tch = torch.from_numpy(self.r_high_np).to(
                device=self.batch_rng.device, dtype=self.batch_rng.dtype)
            self.r_size_tch = torch.from_numpy(self.r_size_np).to(
                device=self.batch_rng.device, dtype=self.batch_rng.dtype)
            
        self.r_low = self.r_low_np if self.lib == 'numpy' else self.r_low_tch
        self.r_size = self.r_size_np if self.lib == 'numpy' else self.r_size_tch
        
        msg_ = f'Some center parameters were left unused: {list(r_params_.keys())}'
        assert len(r_params_) == 0, msg_
        
        ##############################################################
        ############## Plane Normal Sampling Parameters ##############
        ##############################################################
        n_params_ = n_params.copy()

        self.n_dstr = n_dstr
        assert n_dstr in ('uball', 'fixed')
        if n_dstr == 'fixed':
            n_fixed_np = n_params_.pop('value')
            assert n_fixed_np.shape == (n_bch, dim)
            
            n_fixed_size = np.sqrt(np.square(n_fixed_np).sum(axis=-1, keepdims=True))
            self.n_fixed_np = n_fixed_np / n_fixed_size
            
            if self.lib == 'torch':
                self.n_fixed_tch = torch.from_numpy(self.n_fixed_np).to(
                    device=self.batch_rng.device, dtype=self.batch_rng.dtype)
                
            self.n_fixed = self.n_fixed_np if self.lib == 'numpy' else self.n_fixed_tch
        elif n_dstr == 'uball':
            pass
        else:
            raise ValueError(f'n_dstr={n_dstr} undefined')
            
        msg_ = f'Some normal parameters were left unused: {list(n_params_.keys())}'
        assert len(n_params_) == 0, msg_

    def __call__(self, n=1):
        radii = self.r_low + self.r_size * \
            self.batch_rng.uniform((self.n_bch, n))
        radii = radii ** self.r_upow
        
        if self.c_dstr == 'uniform':
            centers = self.batch_rng.uniform((self.n_bch, n, self.d))
            centers = centers * self.c_size + self.c_low
        elif self.c_dstr == 'normal':
            centers = self.batch_rng.normal((self.n_bch, n, self.d))
            centers = centers * self.c_scale + self.c_loc
        elif self.c_dstr == 'ball':
            rnd1 = self.batch_rng.normal((self.n_bch, n, self.d))
            rnd1 = rnd1 / ((rnd1**2).sum(-1, keepdims=True)**0.5)
            
            rnd2 = self.batch_rng.uniform((self.n_bch, n, 1))
            rnd2 = rnd2 ** (1./self.d)
            
            centers = self.c_radi * rnd2 * rnd1 + self.c_cntr
        else:
            raise ValueError(f'c_dstr="{self.c_dstr}" not implemented')
        
        if self.n_dstr == 'uball':   
            rnd3 = self.batch_rng.normal((self.n_bch, n, self.d, 1))
            normals = rnd3 / ((rnd3**2).sum(-2, keepdims=True)**0.5)
        elif self.n_dstr == 'fixed':
            normals = self.n_fixed.reshape(self.n_bch, 1, self.d, 1)
            if self.lib == 'torch':
                normals = normals.expand(self.n_bch, n, self.d, 1)
            elif self.lib == 'numpy':
                normals = np.broadcast_to(normals, 
                    (self.n_bch, n, self.d, 1)).copy()
            else:
                raise ValueError(f'lib={self.lib} not implemented')
        else:
            raise ValueError(f'n_dstr={self.n_dstr} not implemented')
        
        d = dict()
        d['type'] = 'disk'
        d['centers'] = centers
        d['radii'] = radii
        d['normals'] = normals
        return d
    
    

# %% [markdown]
# ### Sruface Sampling

# %% code_folding=[1, 6, 12, 20]
class SphereSampler:
    def __init__(self, batch_rng):
        self.tch_dtype = batch_rng.dtype
        self.tch_device = batch_rng.device
        self.batch_rng = batch_rng

    def np_exlinspace(self, start, end, n):
        assert n >= 1
        a = np.linspace(start, end, n, endpoint=False)
        b = a + 0.5 * (end - a[-1])
        return b

    def tch_exlinspace(self, start, end, n):
        assert n >= 1
        a = torch.linspace(start, end, n+1,
                           device=self.tch_device,
                           dtype=self.tch_dtype)[:-1]
        b = a + 0.5 * (end - a[-1])
        return b

    def __call__(self, volumes, n, do_detspacing=True):
        # volumes -> dictionary
        assert volumes['type'] == 'disk'
        centers = volumes['centers']
        radii = volumes['radii']
        normals = volumes['normals']
        
        n_bch, n_v, d = centers.shape # d is the dimensionality of the original space (usually 3)
        d_n = normals.shape[-1] # The dimensionality of the normal space (usually 1)
        d_k = d - d_n # d_k is the dimensionality of the disk itself (usually 2).
        
        use_np = not torch.is_tensor(centers)
        assert centers.shape == (n_bch, n_v, d)
        assert radii.shape == (n_bch, n_v)
        assert normals.shape == (n_bch, n_v, d, d_n)
        assert not (use_np) or (self.batch_rng.lib == 'numpy')
        assert use_np or (self.batch_rng.device == centers.device)
        assert use_np or (self.batch_rng.dtype == centers.dtype)
        assert self.batch_rng.shape == (n_bch,)
        exlinspace = self.np_exlinspace if use_np else self.tch_exlinspace
        meshgrid = np.meshgrid if use_np else torch.meshgrid
        sin = np.sin if use_np else torch.sin
        cos = np.cos if use_np else torch.cos
        matmul = np.matmul if use_np else torch.matmul

        if do_detspacing and (d_k == 2):
            theta = exlinspace(0.0, 2*np.pi, n)
            assert theta.shape == (n,)
            theta_2d = theta.reshape(n, 1)
            x_tilde_2d_list = [cos(theta_2d), sin(theta_2d)]
            if use_np:
                x_tilde_2d = np.concatenate(x_tilde_2d_list, axis=1)
            else:
                x_tilde_2d = torch.cat(x_tilde_2d_list, dim=1)
            assert x_tilde_2d.shape == (n, d_k)
            x_tilde_4d = x_tilde_2d.reshape(1, 1, n, d_k)
            assert x_tilde_4d.shape == (1, 1, n, d_k)
            x_tilde = x_tilde_4d.expand(n_bch, 1, n, d_k)
            assert x_tilde.shape == (n_bch, 1, n, d_k)
        elif do_detspacing and (d_k == 3):
            n_sqrt = int(np.sqrt(n))
            assert n == n_sqrt * n_sqrt, 'Need n to be int-square for now!'
            theta_1d = exlinspace(0.0, 2*np.pi, n_sqrt)
            unit_unif = exlinspace(0.0, 1.0, n_sqrt)
            if use_np:
                phi_1d = np.arccos(1-2*unit_unif)
            else:
                phi_1d = torch.arccos(1-2*unit_unif)
            theta_msh, phi_msh = meshgrid(theta_1d, phi_1d)
            assert theta_msh.shape == (n_sqrt, n_sqrt)
            assert phi_msh.shape == (n_sqrt, n_sqrt)
            theta_2d, phi_2d = theta_msh.reshape(n, 1), phi_msh.reshape(n, 1)
            assert theta_2d.shape == (n, 1)
            assert phi_2d.shape == (n, 1)
            x_tilde_lst = [sin(phi_2d) * cos(theta),
                           sin(phi_2d) * sin(theta), cos(phi_2d)]
            if use_np:
                x_tilde_2d = np.concatenate(x_tilde_lst, axis=1)
            else:
                x_tilde_2d = torch.cat(x_tilde_lst, dim=1)
            assert x_tilde_2d.shape == (n, d_k)
            x_tilde_4d = x_tilde_2d.reshape(1, 1, n, d_k)
            assert x_tilde_4d.shape == (1, 1, n, d_k)
            x_tilde = x_tilde_4d.expand(n_bch, 1, n, d_k)
            assert x_tilde.shape == (n_bch, 1, n, d_k)
        elif (not do_detspacing) and (not use_np):
            x_tilde_unnorm = self.batch_rng.normal((n_bch, n_v, n, d_k))
            x_tilde_l2 = torch.sqrt(torch.square(x_tilde_unnorm).sum(dim=-1))
            x_tilde = x_tilde_unnorm / x_tilde_l2.reshape(n_bch, n_v, n, 1)
            assert x_tilde.shape == (n_bch, n_v, n, d_k)
        else:
            raise RuntimeError('Not implemented yet!')

        if do_detspacing:
            rot_mats = self.batch_rng.so_n((n_bch, n_v, d_k, d_k))
            assert rot_mats.shape == (n_bch, n_v, d_k, d_k)

        if do_detspacing:
            x_tilde_rot = matmul(x_tilde, rot_mats)
        else:
            x_tilde_rot = x_tilde
        assert x_tilde_rot.shape == (n_bch, n_v, n, d_k)
        
        # Computing a null basis for the plane normals and projecting the 
        # `d_k`-dimensional samples into the original space using the null basis.
        U_n, _, _ = normals.reshape(n_bch, n_v, d, d_n).svd(some=False, compute_uv=True)
        assert U_n.shape == (n_bch, n_v, d, d)
        
        U_tilde_n = U_n[..., d_n:].reshape(n_bch, n_v, 1, d, d_k)
        assert U_tilde_n.shape == (n_bch, n_v, 1, d, d_k)
        
        xtr_ = x_tilde_rot.reshape(n_bch, n_v, n, d_k, 1)
        x_tilde_rp = matmul(U_tilde_n, xtr_).reshape(n_bch, n_v, n, d)
        assert x_tilde_rp.shape == (n_bch, n_v, n, d)

        # Scaling the points to the desired radii and shifting them
        points = x_tilde_rp * \
            radii.reshape(n_bch, n_v, 1, 1) + centers.reshape(n_bch, n_v, 1, d)
        assert points.shape == (n_bch, n_v, n, d)

        if use_np:
            x_tilde_bc = np.broadcast_to(x_tilde, (n_bch, n_v, n, d_k))
        else:
            x_tilde_bc = x_tilde.expand(n_bch, n_v, n, d_k)

        if do_detspacing:
            rot_x_tilde = matmul(x_tilde_bc, rot_mats)
        else:
            rot_x_tilde = x_tilde_bc
        assert rot_x_tilde.shape == (n_bch, n_v, n, d_k)
        
        rxt_ = rot_x_tilde.reshape(n_bch, n_v, n, d_k, 1)
        rp_x_tilde = matmul(U_tilde_n, rxt_).reshape(n_bch, n_v, n, d)
        assert rp_x_tilde.shape == (n_bch, n_v, n, d)
        
        if (d == 3) and (d_n == 1):
            dsknrmls = normals.reshape(n_bch, n_v, 1, d)
            tangents = torch.cross(dsknrmls, rp_x_tilde, dim=-1)
            assert tangents.shape == (n_bch, n_v, n, d)
        else:
            tangents = None

        cst = (2*(np.pi**(d_k/2))) / gamma(d_k/2)
        csts = cst * (radii**(d_k-1))
        assert csts.shape == (n_bch, n_v)

        ret_dict = dict(points=points, normals=rp_x_tilde, tangents=tangents, areas=csts)
        return ret_dict



# %% [markdown]
# ### Visualization Utility Functions

# %%
def create_xg(xi_low, xi_high, n_gi, tch_device, tch_dtype):
    dim = len(xi_low)
    xi_1d_list = []
    for i_dim, ng_dim in enumerate(n_gi):
        xi_1d = torch.linspace(xi_low[i_dim], xi_high[i_dim], 
            ng_dim, dtype=tch_dtype, device=tch_device)
        xi_1d_list.append(xi_1d)
        
    n_g = np.prod(n_gi)
    x_g = torch.cartesian_prod(*xi_1d_list)
    assert x_g.shape == (n_g, dim)
    
    return x_g.reshape(*n_gi, dim)

def draw_heatmap(x_g, y_g, fig, axes, share_cnorm=True, 
    cmap='RdBu', field_abrv='A', components='xyz', 
    show_title='all', zstr_loc='top'):
    
    n_g1, n_g2, n_g3, dim = x_g.shape
    x3_1d = x_g[0, 0, :, 2]
    n_g = n_g1 * n_g2 * n_g3
    
    assert show_title in ('all', 'top', 'none')
    assert zstr_loc in ('top', 'left')
    
    i_comps = ['xyz'.index(ss) for ss in components]
    n_comps = len(components)
    
    assert dim == 3
    assert y_g.shape == (n_g, dim)

    y_g2 = y_g.reshape(n_g1, n_g2, n_g3, dim).detach().cpu().numpy()
    assert y_g2.shape == (n_g1, n_g2, n_g3, dim)

    x_g2 = x_g.reshape(n_g1, n_g2, n_g3, dim).detach().cpu().numpy()
    assert x_g2.shape == (n_g1, n_g2, n_g3, dim)

    cmap = mpl.cm.get_cmap(cmap)
    cnorm = mpl.colors.Normalize(vmin=y_g.min().item(), vmax=y_g.max().item())
    cnorm = cnorm if share_cnorm else None
    
    axes = np.array(axes)
    
    n_q = int(axes.size // n_comps)
    q = np.linspace(0, 1, n_q)
    zi_list = np.quantile(np.arange(n_g3), q).round().astype(int).tolist()
    
    for ax_idx, ax in enumerate(axes.reshape(-1)):
        di = i_comps[ax_idx %  n_comps]
        
        # zi = ax_idx // n_comps
        zi = zi_list[ax_idx // n_comps]
        
        if ax_idx >= (n_g3 * n_comps):
            break
        
        x_hmap = x_g2[:, :, zi, :]
        assert x_hmap.shape == (n_g1, n_g2, 3)
        assert (x_hmap[:, :, 2] == x_hmap[0, 0, 2]).all()
        
        y_hmap = y_g2[:, :, zi, di]
        assert y_hmap.shape == (n_g1, n_g2)
        
        im = ax.pcolormesh(x_hmap[:, :, 0], x_hmap[:, :, 1], y_hmap, 
            shading='auto', norm=cnorm, cmap=cmap, linewidth=0, 
            rasterized=True)
        
        n_rows, n_cols = axes.shape
        row, col = ax_idx // n_cols, ax_idx % n_cols
        z_str = f' ($z$={x3_1d[zi].item():.2f})'
        
        x_label, y_label = '$x$', '$y$'
        ax_title = ''
        if field_abrv is not None:
            ax_title += f'${field_abrv}_{"xyz"[di]}$'
        if zstr_loc == 'top':
            ax_title += z_str
        elif zstr_loc == 'left':
            y_label += z_str
        else:
            raise ValueError(f'zstr_loc={zstr_loc} undefined')
        
        if row == (n_rows - 1):
            ax.set_xlabel(x_label)
        else:
            ax.set_xticks([])
        
        if col == 0:
            ax.set_ylabel(y_label)
        else:
            ax.set_yticks([])
        
        print_title  = (show_title == 'all')
        print_title += (show_title == 'top') and (row == 0)
        print_title += (show_title == 'none') and False
        if print_title:
            ax.set_title(ax_title)
    
    if show_title in ('top', 'none'): 
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

def draw_quiver(x_g, y_g, fig, axes, share_cnorm=True, 
    cmap='RdBu', field_abrv='A', show_title='all', 
    zstr_loc='top', subsamp=1, **kwargs):
    
    n_g1, n_g2, n_g3, dim = x_g.shape
    x3_1d = x_g[0, 0, :, 2]
    n_g = n_g1 * n_g2 * n_g3
    
    assert dim == 3
    assert y_g.shape == (n_g, dim)

    y_g2 = y_g.reshape(n_g1, n_g2, n_g3, dim).detach().cpu().numpy()
    assert y_g2.shape == (n_g1, n_g2, n_g3, dim)

    x_g2 = x_g.reshape(n_g1, n_g2, n_g3, dim).detach().cpu().numpy()
    assert x_g2.shape == (n_g1, n_g2, n_g3, dim)

    cmap = mpl.cm.get_cmap(cmap)
    cnorm = mpl.colors.Normalize(vmin=y_g[:, 2].min().item(), vmax=y_g[:, 2].max().item())
    cnorm = cnorm if share_cnorm else None
    
    axes = np.array(axes)
    n_q = int(axes.size)
    q = np.linspace(0, 1, n_q)
    zi_list = np.quantile(np.arange(n_g3), q).round().astype(int).tolist()
    
    for ax_idx, ax in enumerate(axes.reshape(-1)):
        # zi = ax_idx
        zi = zi_list[ax_idx]
        
        x_hmap = x_g2[:, :, zi, :]
        assert x_hmap.shape == (n_g1, n_g2, dim)
        assert (x_hmap[:, :, 2] == x_hmap[0, 0, 2]).all()
        
        y_hmap = y_g2[:, :, zi, :]
        assert y_hmap.shape == (n_g1, n_g2, dim)
        
        ss = subsamp
        ax.quiver(x_hmap[::ss, ::ss, 0], x_hmap[::ss, ::ss, 1], 
            y_hmap[::ss, ::ss, 0], y_hmap[::ss, ::ss, 1], y_hmap[::ss, ::ss, 2], 
            norm=cnorm, cmap=cmap, **kwargs)
        
        n_rows, n_cols = axes.shape
        row, col = ax_idx // n_cols, ax_idx % n_cols
        z_str = f' ($z$={x3_1d[zi].item():.2f})'
        
        x_label, y_label = '$x$', '$y$'
        ax_title = ''
        if field_abrv is not None:
            ax_title += f'${field_abrv}$'
        if zstr_loc == 'top':
            ax_title += z_str
        elif zstr_loc == 'left':
            y_label += z_str
        else:
            raise ValueError(f'zstr_loc={zstr_loc} undefined')
        
        if row == (n_rows - 1):
            ax.set_xlabel(x_label)
        else:
            ax.set_xticks([])
        
        if col == 0:
            ax.set_ylabel(y_label)
        else:
            ax.set_yticks([])
        
        print_title  = (show_title == 'all')
        print_title += (show_title == 'top') and (row == 0)
        print_title += (show_title == 'none') and False
        if print_title:
            ax.set_title(ax_title)
    
    if show_title in ('top', 'none'): 
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        


# %% [markdown]
# ### Mathematical Utility Functions

# %%
def curl(A, x, create_graph=True):
    assert A.shape[-1] == 3
    assert x.shape[-1] == 3
    assert A.shape == x.shape
    
    Ax, Ay, Az = A[..., 0], A[..., 1], A[..., 2]
    Ax_, Ay_, Az_ = Ax.sum(), Ay.sum(), Az.sum()

    Az_xyz, = torch.autograd.grad(Az_, [x],
        grad_outputs=None, retain_graph=True, create_graph=create_graph,
        only_inputs=True, allow_unused=False)
    Ay_xyz, = torch.autograd.grad(Ay_, [x],
        grad_outputs=None, retain_graph=True, create_graph=create_graph,
        only_inputs=True, allow_unused=False)
    Ax_xyz, = torch.autograd.grad(Ax_, [x],
        grad_outputs=None, retain_graph=True, create_graph=create_graph,
        only_inputs=True, allow_unused=False)

    Az_x, Az_y = Az_xyz[..., 0], Az_xyz[..., 1]
    Ay_x, Ay_z = Ay_xyz[..., 0], Ay_xyz[..., 2]
    Ax_y, Ax_z = Ax_xyz[..., 1], Ax_xyz[..., 2]

    A_curl = torch.stack([Az_y - Ay_z, Ax_z - Az_x, Ay_x - Ax_y], dim=-1)
    return A_curl


# MCMC Utility Functions
class MCMCWireEncoder:
    def __init__(self, kind, source, sink, current):
        assert isinstance(source, np.ndarray)
        assert isinstance(sink, np.ndarray)
        assert isinstance(current, np.ndarray)
        source = source.copy()
        sink = sink.copy()
        current = current.copy()
        n_seeds, wire_n, dim = source.shape
        
        if kind == 'srcsnkxy':
            assert dim == 3
            sdim = wire_n * 2
            source = source.copy()
            source[:, :, :2] = np.nan
            sink[:, :, :2] = np.nan
        else:
            raise ValueError(f'kind={kind} not implmntd')
        
        self.kind = kind
        self.n_seeds = n_seeds
        self.wire_n = wire_n
        self.dim = dim
        self.sdim = sdim
        self.source = source
        self.sink = sink
        self.current = current
        
    def __call__(self, mu):
        kind, n_seeds = self.kind, self.n_seeds
        wire_n, dim, sdim = self.wire_n, self.dim, self.sdim
        source, sink, current = self.source, self.sink, self.current
        
        assert mu.shape == (n_seeds, sdim)
        
        if kind == 'srcsnkxy':
            src_enc = source.copy()
            src_enc[:, :, :2] = mu.reshape(n_seeds, wire_n, 2)
            assert src_enc.shape == (n_seeds, wire_n, dim)
            
            snk_enc = sink.copy()
            snk_enc[:, :, :2] = mu.reshape(n_seeds, wire_n, 2)
            assert snk_enc.shape == (n_seeds, wire_n, dim)
            
            cur_enc = current.copy()
            assert cur_enc.shape == (n_seeds, wire_n)
        else:
            raise ValueError(f'kind={kind} not defined')
        
        outdict = dict(source=src_enc, sink=snk_enc, current=cur_enc)
        return outdict            


def normal_logprob(x, loc, scale):
    n_seeds, n_pnts, dim = loc.shape
    assert scale.shape == (n_seeds, n_pnts)
    
    scale2 = scale.reshape(n_seeds, n_pnts, 1)
    assert scale2.shape == (n_seeds, n_pnts, 1)
    
    e = (x - loc) / scale2
    assert e.shape == (n_seeds, n_pnts, dim)
    
    hltp = np.log(2*np.pi).item() / 2
    ll = -0.5 * e.square() - hltp - scale2.log()
    assert ll.shape == (n_seeds, n_pnts, dim)
    
    llsum = ll.sum(dim=-1)
    assert llsum.shape == (n_seeds, n_pnts)
    
    return llsum


def replicate_top(param, top_idx):
    n_seeds = param.shape[0]
    n_grps, srch_reset_k = top_idx.shape
    
    n_cpg = n_seeds // n_grps
    assert n_seeds == (n_grps * n_cpg)
    
    srch_reset_reps = n_cpg // srch_reset_k
    assert n_cpg == (srch_reset_reps * srch_reset_k)
    
    pshape = param.shape[1:]
    opshape = tuple([1 for _ in pshape])
    
    param2 = param.reshape(n_grps, n_cpg, *pshape)
    assert param2.shape == (n_grps, n_cpg, *pshape)
    
    top_idx2 = top_idx.reshape(n_grps, srch_reset_k, *opshape)
    assert top_idx2.shape == (n_grps, srch_reset_k, *opshape)
    
    param3 = torch.take_along_dim(param2, top_idx2, dim=1)
    assert param3.shape == (n_grps, srch_reset_k, *pshape)
    
    param4 = param3.unsqueeze(1)
    assert param4.shape == (n_grps, 1, srch_reset_k, *pshape)
    
    param5 = param4.expand(n_grps, srch_reset_reps, srch_reset_k, *pshape).clone()
    assert param5.shape == (n_grps, srch_reset_reps, srch_reset_k, *pshape)
    
    param6 = param5.reshape(n_seeds, *pshape)
    assert param6.shape == (n_seeds, *pshape)
    
    return param6



# %% [markdown]
# ### Visualization

# %% code_folding=[0, 57, 66, 142, 225]
def get_nn_sol(model, x, n_eval=None, get_field=True, 
    out_lib='numpy'):
    """
    Gets a model and evaluates it minibatch-wise on the tensor x. 
    The minibatch size is capped at n_eval. The output will have the 
    predicted potentials and the vector fields at them.

    Parameters
    ----------
    model: (nn.module) the batched neural network.

    x: (torch.tensor) the evaluation points. This array should be 
        >2-dimensional and have a shape of `(..., x_rows, x_cols)`.

    n_eval: (int or None) the maximum mini-batch size. If None is 
        given, `x_rows` will be used as `n_eval`.
        
    out_lib: (str) determines the output tensor type. Should be either 
        'numpy' or 'torch'.
    
    Output Dictionary
    ----------
    v: (np.array or torch.tensor) the evaluated potentials 
        with a shape of `(*model.shape, x_rows)` where
        model.shape is the batch dimensions of the model. 

    e: (np.array or torch.tensor) the evaluated vector fields 
        with a shape of `(*model.shape, x_rows, x_cols)` where
        model.shape is the batch dimensions of the model.
    """
    x_rows, x_cols = tuple(x.shape)[-2:]
    x_bd_ = tuple(x.shape)[:-2]
    x_bd = (1,) if len(x_bd_) == 0 else x_bd_
    msg_ = f'Cannot have {x.shape} fed to {model.shape}'
    assert len(x_bd) <= model.ndim, msg_
    if len(x_bd) < model.ndim:
        x_bd = tuple([1] * (model.ndim-len(x_b)) + list(x_bd))
    assert all((a == b) or (a == 1) or (b == 1) 
               for a, b in zip(x_bd, model.shape)), msg_
    n_eval = x_rows if n_eval is None else n_eval
    if out_lib == 'numpy':
        to_lib = lambda a: a.detach().cpu().numpy()
        lib_cat = lambda al: np.concatenate(al, axis=1)
        lpf = '_np'
    elif out_lib == 'torch':
        to_lib = lambda a: a
        lib_cat = lambda al: torch.cat(al, dim=1)
        lpf = ''
    else:
        raise ValueError(f'outlib={outlib} not defined.')

    n_batches = int(np.ceil(x_rows / n_eval))
    v_pred_list = []
    e_pred_list = []
    for i in range(n_batches):
        x_i = x[..., (i*n_eval):((i+1)*n_eval), :]
        xi_rows = x_i.shape[-2]
        x_ii = x_i.reshape(*x_bd, xi_rows, x_cols)
        x_iii = x_ii.expand(*model.shape, xi_rows, x_cols)
        x_iiii = nn.Parameter(x_iii)
        v_pred_i = model(x_iiii)
        v_pred_ii = to_lib(v_pred_i.detach())
        v_pred_list.append(v_pred_ii)
        if get_field:
            e_pred_i = curl(v_pred_i, x_iiii, create_graph=False)
            e_pred_ii = to_lib(e_pred_i.squeeze(-1).detach())
            e_pred_list.append(e_pred_ii)

    v_pred = lib_cat(v_pred_list)
    if get_field:
        e_pred = lib_cat(e_pred_list)
    else:
        e_pred = None

    outdict = {f'v{lpf}': v_pred, f'e{lpf}': e_pred}
    return outdict


def get_prob_sol(problem, x, n_eval=None, get_field=True, 
    out_lib='numpy'):
    """
    Gets a problem and evaluates the analytical solution to its 
    potentials and vector fields minibatch-wise on the tensor x. 
    The minibatch size is capped at n_eval. The output will have the 
    predicted potentials and the vector fields at them.

    Parameters
    ----------
    problem: (object) the problem with both the `potential` and 
        `field` methods for analytical solution evaluation.

    x: (torch.tensor) the evaluation points. This array should be 
        >2-dimensional and have a shape of `(..., x_rows, x_cols)`.

    n_eval: (int or None) the maximum mini-batch size. If None is 
        given, `x_rows` will be used as `n_eval`.

    Output Dictionary
    ----------
    v_np: (np.array) the evaluated potentials with a shape of
        `(..., x_rows)`. 

    e_np: (np.array) the evaluated vector fields with a shape of
        `(..., x_rows, x_cols)`.
    """

    assert hasattr(problem, 'potential')
    assert callable(problem.potential)
    assert hasattr(problem, 'field')
    assert callable(problem.field)

    x_rows, x_cols = tuple(x.shape)[-2:]
    x_bd_ = tuple(x.shape)[:-2]
    x_bd = (1,) if len(x_bd_) == 0 else x_bd_
    msg_ = f'Cannot have {x.shape} fed to {problem.shape}'
    assert len(x_bd) <= problem.ndim, msg_
    if len(x_bd) < problem.ndim:
        x_bd = tuple([1] * (problem.ndim-len(x_b)) + list(x_bd))
    assert all((a == b) or (a == 1) or (b == 1) 
               for a, b in zip(x_bd, problem.shape)), msg_
    n_eval = x_rows if n_eval is None else n_eval
    if out_lib == 'numpy':
        to_lib = lambda a: a.detach().cpu().numpy()
        lib_cat = lambda al: np.concatenate(al, axis=1)
        lpf = '_np'
    elif out_lib == 'torch':
        to_lib = lambda a: a
        lib_cat = lambda al: torch.cat(al, dim=1)
        lpf = ''
    else:
        raise ValueError(f'outlib={outlib} not defined.')

    n_batches = int(np.ceil(x_rows / n_eval))
    v_list = []
    e_list = []
    for i in range(n_batches):
        x_i = x[..., (i*n_eval):((i+1)*n_eval), :]
        xi_rows = x_i.shape[-2]
        x_ii = x_i.reshape(*x_bd, xi_rows, x_cols)
        x_iii = x_ii.expand(*problem.shape, xi_rows, x_cols)
        v_i = problem.potential(x_iii)
        v_list.append(to_lib(v_i))
        if get_field:
            e_i = problem.field(x_iii)
            e_list.append(to_lib(e_i))

    v = lib_cat(v_list)
    if get_field:
        e = lib_cat(e_list)
    else:
        e = None
    outdict = {f'v{lpf}': v, f'e{lpf}': e}
    return outdict

# useless!
def make_grid(x_low, x_high, dim, n_gpd, lib):
    """
    Creates a grid of points using the mesgrid functions
    
    Parameters
    ----------
    x_low: (list) a list of length `dim` with floats 
        representing the lower limits of the grid.
    
    x_high: (list) a list of length `dim` with floats 
        representing the higher limits of the grid.
    
    dim: (int) the dimension of the grid space.
    
    n_gpd: (int) the number of points in each 
        grid dimension. This yields a total of 
        `n_gpd**dim` points in the total grid.
        
    lib: (str) either 'torch' or 'numpy'. This determines 
        the type of `x` output.
        
    Outputs
    -------
    x: (torch.tensor or np.array) a 2-d tensor or array 
        with the shape of `(n_gpd**dim, dim)`. 
    
    xi_msh_np: (list of np.array) a list of length `dim` 
        with meshgrid tensors each with a shape of 
        `[n_gpd] * dim`.
    """
    
    assert dim == 2, 'not implemented yet'
    assert len(x_low) == dim
    assert len(x_high) == dim
    assert lib in ('torch', 'numpy')
    library = torch if lib == 'torch' else np
    tnper = lambda a: a.cpu().detach().numpy()
    nper = tnper if lib == 'torch' else lambda a: a
    
    x1_low, x2_low = x_low
    x1_high, x2_high = x_high
    n_g_plt = n_gpd ** dim

    x1_1d = library.linspace(x1_low, x1_high, n_gpd)
    assert x1_1d.shape == (n_gpd,)

    x2_1d = library.linspace(x2_low, x2_high, n_gpd)
    assert x2_1d.shape == (n_gpd,)

    x1_msh, x2_msh = library.meshgrid(x1_1d, x2_1d)
    assert x1_msh.shape == (n_gpd, n_gpd)
    assert x2_msh.shape == (n_gpd, n_gpd)

    x1 = x1_msh.reshape(n_g_plt, 1)
    assert x1.shape == (n_g_plt, 1)

    x2 = x2_msh.reshape(n_g_plt, 1)
    assert x2.shape == (n_g_plt, 1)

    x1_1d_c = x1_1d.reshape(n_gpd, 1)
    assert x1_1d_c.shape == (n_gpd, 1)

    x2_1d_c = x2_1d.reshape(n_gpd, 1)
    assert x2_1d_c.shape == (n_gpd, 1)

    x1_msh_np = nper(x1_msh)
    assert x1_msh_np.shape == (n_gpd, n_gpd)

    x2_msh_np = nper(x2_msh)
    assert x1_msh_np.shape == (n_gpd, n_gpd)

    x = torch.cat([x1, x2], dim=1)
    assert x.shape == (n_g_plt, dim)

    x_np = nper(x)
    assert x_np.shape == (n_g_plt, dim)
    
    xi_msh_np = [x1_msh_np, x2_msh_np]
    outdict = dict(x=x, xi_msh_np=xi_msh_np)

    return outdict


def plot_sol(x1_msh_np, x2_msh_np, sol_dict, fig=None, ax=None, cax=None):
    n_gpd, dim = x1_msh_np.shape[0], x1_msh_np.ndim
    assert dim == 2, f'dim={dim}, x1_msh_np.shape={x1_msh_np.shape}'
    assert x1_msh_np.shape == (n_gpd, n_gpd)
    assert x2_msh_np.shape == (n_gpd, n_gpd)
    n_g = (n_gpd ** dim)
   
    if fig is None:
        assert ax is None
        assert cax is None
        fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.5), dpi=72)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
    else:
        assert ax is not None
   
    e_percentile_cap = 90
    if 'v_np' in sol_dict:
        v_np = sol_dict['v_np']
    else:
        v_np = sol_dict['v'].detach().cpu().numpy()
    assert v_np.shape[-1] == n_g
    
    v_msh_np = v_np.reshape(-1, n_gpd, n_gpd).mean(axis=0)
    im = ax.pcolormesh(x1_msh_np, x2_msh_np, v_msh_np,
                        shading='auto', cmap='RdBu')
    if cax is not None:
        fig.colorbar(im, cax=cax)

    if 'e_np' in sol_dict:
        e_msh_np = sol_dict['e_np']
    else:
        e_msh_np = sol_dict['e']
        if e_msh_np is not None:
            e_msh_np = e_msh_np.detach().cpu().numpy()
    
    if e_msh_np is not None:
        assert e_msh_np.shape[-2:] == (n_g, dim)
        e_msh_np = e_msh_np.reshape(-1, n_gpd,
            n_gpd, dim).mean(axis=0)
        if e_percentile_cap is not None:
            e_size = np.sqrt((e_msh_np**2).sum(axis=-1))
            e_size_cap = np.percentile(a=e_size, 
                q=e_percentile_cap, axis=None)
            cap_coef = np.ones_like(e_size)
            cap_coef[e_size > e_size_cap] = e_size_cap / \
                e_size[e_size > e_size_cap]
            e_msh_capped = e_msh_np * \
                cap_coef.reshape(*e_msh_np.shape[:-1], 1)
        else:
            e_msh_capped = e_msh_np

        ax.quiver(x1_msh_np, x2_msh_np,
            e_msh_capped[:, :, 0], e_msh_capped[:, :, 1])
    return fig, ax, cax

def get_perfdict(e_pnts, e_mdlsol, e_prbsol):
    """
    Computes the biased, bias-corrected, and slope-corrected error 
    metrics for the solutions of a Poisson problem.
    
    This function computes three types of MSE and MAE statistics:
        
        1. Plain: just take the model and ground truth solution
            and subtract them to get the errors. No bias- or slope-correction 
            is applied to offset those degrees of freedom.
            
            shorthand: 'pln'
            
        2. Bias-corrected: subtracts the average value from both the model 
            and ground truth solutions, and then computes the errors.
            
            shorthand: 'bc'
            
        3. Slope-corrected: Since any linear function can be added to the
            Poisson solutions without violating the poisson equation, this
            function fits an ordinary least squares to both the model and
            ground truth solutions, and then subtracts it from them. This
            way, even the arbitrary-slope issue can be addressed.
            
            shorthand: 'slc'
            
    Parameters
    ----------
    e_pnts: (torch.tensor) The input points to the model and the ground truth.
        This should have a shape of (n_seeds, n_evlpnts, dim).
        
    e_mdlsol: (torch.tensor) The model solution with a
        (n_seeds, n_evlpnts) shape.
    
    e_prbsol: (torch.tensor) The ground truth solution with a
        (n_seeds, n_evlpnts) shape.
        
    Output
    ------
    outdict: (dict) A mapping between the error keys and their numpy arrays.
        The error keys are the cartesian product of ('pln', 'bc', 'slc') 
        and ('mse', 'mae').
    """
    n_seeds, n_evlpnts, dim = e_pnts.shape
    assert e_mdlsol.shape == (n_seeds, n_evlpnts, dim)
    assert e_prbsol.shape == (n_seeds, n_evlpnts, dim)
    
    with torch.no_grad():
        # The plain non-processed error matrix
        err_pln = e_mdlsol - e_prbsol
        assert err_pln.shape == (n_seeds, n_evlpnts, dim)
        
        # The bias-corrected error matrix
        e_mdlsol2 = e_mdlsol - e_mdlsol.mean(dim=1, keepdim=True)
        assert e_mdlsol2.shape == (n_seeds, n_evlpnts, dim)
        e_prbsol2 = e_prbsol - e_prbsol.mean(dim=1, keepdim=True)
        assert e_prbsol2.shape == (n_seeds, n_evlpnts, dim)
        err_bc = e_mdlsol2 - e_prbsol2
        assert err_bc.shape == (n_seeds, n_evlpnts, dim)
        
        # The normalized error matrix by std of all dimensions
        e_mdlsol3 = e_mdlsol2.reshape(n_seeds, n_evlpnts*dim)
        assert e_mdlsol3.shape == (n_seeds, n_evlpnts*dim)
        
        e_prbsol3 = e_prbsol2.reshape(n_seeds, n_evlpnts*dim)
        assert e_prbsol3.shape == (n_seeds, n_evlpnts*dim)
        
        e_mdlsol4 = e_mdlsol3 / e_mdlsol3.std(dim=1, keepdim=True)
        assert e_mdlsol4.shape == (n_seeds, n_evlpnts*dim)
        
        e_prbsol4 = e_prbsol3 / e_prbsol3.std(dim=1, keepdim=True)
        assert e_prbsol4.shape == (n_seeds, n_evlpnts*dim)
        
        err_scn = e_mdlsol4 - e_prbsol4
        assert err_scn.shape == (n_seeds, n_evlpnts*dim)
        
        # The normalized error matrix individually for each dim
        e_mdlsol5 = e_mdlsol2 / e_mdlsol2.square().mean(dim=1, keepdim=True).sqrt()
        assert e_mdlsol5.shape == (n_seeds, n_evlpnts, dim)
        
        e_prbsol5 = e_prbsol2 / e_prbsol2.square().mean(dim=1, keepdim=True).sqrt()
        assert e_prbsol5.shape == (n_seeds, n_evlpnts, dim)
        
        err_bcn = e_mdlsol5 - e_prbsol5
        assert err_bcn.shape == (n_seeds, n_evlpnts, dim)
        
        # Computing the mse and mae values
        e_plnmse = err_pln.square().sum(dim=-1).mean(dim=-1)
        assert e_plnmse.shape == (n_seeds,)
        e_plnmae = err_pln.abs().sum(dim=-1).mean(dim=-1)
        assert e_plnmse.shape == (n_seeds,)
        
        e_bcmse = err_bc.square().sum(dim=-1).mean(dim=-1)
        assert e_bcmse.shape == (n_seeds,)
        e_bcmae = err_bc.abs().sum(dim=-1).mean(dim=-1)
        assert e_bcmse.shape == (n_seeds,)
        
        e_scnmse = err_scn.square().mean(dim=-1)
        assert e_scnmse.shape == (n_seeds,)
        e_scnmae = err_scn.abs().mean(dim=-1)
        assert e_scnmse.shape == (n_seeds,)
        
        e_bcnmse = err_bcn.square().mean(dim=[-1, -2])
        assert e_bcnmse.shape == (n_seeds,)
        e_bcnmae = err_bcn.abs().mean(dim=[-1, -2])
        assert e_bcnmse.shape == (n_seeds,)
    
        outdict = {'pln/mse': e_plnmse.detach().cpu().numpy(),
                   'pln/mae': e_plnmae.detach().cpu().numpy(),
                   'bc/mse': e_bcmse.detach().cpu().numpy(),
                   'bc/mae': e_bcmae.detach().cpu().numpy(),
                   'bcn/mse': e_bcnmse.detach().cpu().numpy(),
                   'bcn/mae': e_bcnmae.detach().cpu().numpy(),
                   'scn/mse': e_scnmse.detach().cpu().numpy(),
                   'scn/mae': e_scnmae.detach().cpu().numpy()}
    
    return outdict



# %% [markdown]
# ## Optional Visualization Tests

# %% [markdown]
# ### Visualizing the True Potential and Fields

# %% tags=["active-ipynb"]
# ex_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# ex_tchdevice = torch.device(ex_device)
# ex_tchdtype = torch.double
#
# n_bch = 1
# ex_rng = BatchRNG(shape=(n_bch,), lib='torch', device=ex_tchdevice, dtype=ex_tchdtype,
#                unif_cache_cols=10000, norm_cache_cols=10000)
# ex_rng.seed(np.broadcast_to(12345+np.arange(n_bch), ex_rng.shape))

# %% tags=["active-ipynb"]
# dim = 3
# n_wires = 3
# ex_source = [[[-0.5, -0.5, -1], [0, 0, -1], [0.5, 0.5, -1]]]
# ex_sink =   [[[-0.5, -0.5,  1], [0, 0,  1], [0.5, 0.5,  1]]]
# ex_source = np.broadcast_to(ex_source, (n_bch, n_wires, dim)).copy()
# ex_sink = np.broadcast_to(ex_sink, (n_bch, n_wires, dim)).copy()
# current = np.ones((n_bch, n_wires))
# # ex_source, ex_sink = ex_source + 1e-3, ex_sink + 1e-3
# ex_problem = DeltaLineProblem(source=ex_source, sink=ex_sink, current=current,
#     tch_device=ex_tchdevice, tch_dtype=ex_tchdtype)

# %% tags=["active-ipynb"]
# ex_ng1, ex_ng2, ex_ng3 = (200, 200, 6)
# ex_xilow  = [-1., -1., 0.]
# ex_xihigh = [ 1.,  1., 1.]
# ex_ngi = (ex_ng1, ex_ng2, ex_ng3)
#
# ex_dim, ex_n_xg = len(ex_ngi), int(np.prod(ex_ngi))
#
# ex_x_g = create_xg(ex_xilow, ex_xihigh, ex_ngi, ex_tchdevice, ex_tchdtype)
# assert ex_x_g.shape == (ex_ng1, ex_ng2, ex_ng3, ex_dim)
#
# ex_x_gf = ex_x_g.reshape(1, ex_n_xg, ex_dim).expand(n_bch, ex_n_xg, ex_dim)
# assert ex_x_gf.shape == (n_bch, ex_n_xg, ex_dim)
#
# # 1. The magnetic potentials
# A_gf1 = ex_problem.potential(ex_x_gf)
# assert A_gf1.shape == (n_bch, ex_n_xg, ex_dim)
#
# A_gf2 = A_gf1.clip(A_gf1.nanquantile(0.001, dim=1, keepdim=True), 
#                    A_gf1.nanquantile(0.999, dim=1, keepdim=True))
# assert A_gf2.shape == (n_bch, ex_n_xg, ex_dim)
#
# y_g1 = A_gf2.mean(dim=0)
# assert y_g1.shape == (ex_n_xg, ex_dim)
#
# # 2. The magnetic fields (1)
# B_gf1 = ex_problem.field(ex_x_gf)
# assert B_gf1.shape == (n_bch, ex_n_xg, ex_dim)
#
# B_gf2 = B_gf1.clip(B_gf1.nanquantile(0.01, dim=1, keepdim=True), 
#                    B_gf1.nanquantile(0.99, dim=1, keepdim=True))
# assert B_gf2.shape == (n_bch, ex_n_xg, ex_dim)
#
# y_g2 = B_gf2.mean(dim=0)
# assert y_g2.shape == (ex_n_xg, ex_dim)
#
# # 3. The magnetic field (2)
# B_gf1n = B_gf1.norm(dim=-1, keepdim=True)
# assert B_gf1n.shape == (n_bch, ex_n_xg, 1)
#
# B_gf2n = B_gf1n.clip(B_gf1n.nanquantile(0.1, dim=1, keepdim=True), 
#                      B_gf1n.nanquantile(0.9, dim=1, keepdim=True))
# assert B_gf2n.shape == (n_bch, ex_n_xg, 1)
#
# B_gf3 = B_gf1 * (B_gf2n / B_gf1n)
# assert B_gf3.shape == (n_bch, ex_n_xg, ex_dim)
#
# y_g3 = B_gf3.mean(dim=0)
# assert y_g3.shape == (ex_n_xg, ex_dim)

# %% [markdown]
# ### Checking the Analytical Curl Property 

# %% tags=["active-ipynb"]
# ex_nx = 2500
# ex_x = ex_rng.uniform((n_bch, ex_nx, dim)) * 2 - 1.
# ex_x = torch.nn.Parameter(ex_x)
# assert ex_x.shape == (n_bch, ex_nx, dim)
#
# ex_A = ex_problem.potential(ex_x)
# assert ex_A.shape == (n_bch, ex_nx, dim)
#
# ex_B = ex_problem.field(ex_x)
# assert ex_B.shape == (n_bch, ex_nx, dim)
#
# ex_Acurl = curl(ex_A, ex_x)
# assert ex_Acurl.shape == (n_bch, ex_nx, dim)
#
# assert (ex_B - ex_Acurl).abs().max() < 1e-4

# %% [markdown]
# ### Visualizing the Magnetic Potential and Field Components

# %% tags=["active-ipynb"]
# n_rows, n_cols = 2, 3 
# ex1_fig, ex1_axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2., n_rows*2.), 
#     sharex=False, sharey=False)
# ex1_axes = np.array(ex1_axes).reshape(n_rows, n_cols)
# draw_heatmap(ex_x_g, y_g1, ex1_fig, ex1_axes, field_abrv='A', 
#     components='xyz', show_title='top', zstr_loc='left')
# ex1_fig

# %% tags=["active-ipynb"]
# n_rows, n_cols = ex_ngi[-1]//3, 3
# ex2_fig, ex2_axes = plt.subplots(n_rows, n_cols, 
#     figsize=(n_cols*2., n_rows*2.), sharex=False, sharey=False)
# ex2_axes = np.array(ex2_axes).reshape(n_rows, n_cols)
# draw_heatmap(ex_x_g, y_g1, ex2_fig, ex2_axes, field_abrv='A', 
#     components='z', show_title='all')
# ex2_fig

# %% tags=["active-ipynb"]
# n_rows, n_cols = ex_ngi[-1]//3, 3
# ex3_fig, ex3_axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2., n_rows*2.), 
#     sharex=False, sharey=False, dpi=100)
# ex3_axes = np.array(ex3_axes).reshape(n_rows, n_cols)
# draw_quiver(ex_x_g, y_g3, ex3_fig, ex3_axes, field_abrv='B', 
#     show_title='all', cmap='RdBu', subsamp=10)
# ex3_fig

# %% [markdown]
# ### Visualizing Example Sampled Volumes and Surface Tangents

# %% tags=["active-ipynb"]
# n_bch = 5
# ex4_rng = BatchRNG(shape=(n_bch,), lib='torch', device=ex_tchdevice, dtype=ex_tchdtype,
#                unif_cache_cols=5000, norm_cache_cols=5000)
# ex4_rng.seed(np.broadcast_to(12345+np.arange(n_bch), ex4_rng.shape))
#
# dim = 3
# ex4_nwires = 3
# ex4_source = [[[-0.5, -0.5, -1], [0, 0, -1], [0.5, 0.5, -1]]]
# ex4_sink =   [[[-0.5, -0.5,  1], [0, 0,  1], [0.5, 0.5,  1]]]
# ex4_source = np.broadcast_to(ex4_source, (n_bch, ex4_nwires, dim)).copy()
# ex4_sink = np.broadcast_to(ex4_sink, (n_bch, ex4_nwires, dim)).copy()
# ex4_current = np.ones((n_bch, ex4_nwires))
# prob_ex4 = DeltaLineProblem(source=ex4_source, sink=ex4_sink, 
#     current=ex4_current, tch_device=ex_tchdevice, tch_dtype=ex_tchdtype)
#
# volsampler_ex4 = DiskSampler(c_dstr='uniform', c_params=dict(
#                              low=np.broadcast_to([-1.0, -1.0, -1.0], (n_bch, 3)).copy(),
#                              high=np.broadcast_to([1.0,  1.0,  1.0], (n_bch, 3)).copy() ),
#                              r_dstr='uniform', r_params=dict(
#                              low=np.broadcast_to([0.1], (n_bch,)).copy(),
#                              high=np.broadcast_to([1.5], (n_bch,)).copy()),
#                              n_dstr='uball', n_params=dict(),
#                              batch_rng=ex4_rng)
#
# sphsampler_ex4 = SphereSampler(batch_rng=ex4_rng)
#
# ex4_nvol, ex4_nsrfpts, ex4_dim = 5, 100, 3
# ex4_vols = volsampler_ex4(n=ex4_nvol)
# ex4_cntrvols = ex4_vols['centers']
# assert ex4_cntrvols.shape == (n_bch, ex4_nvol, ex4_dim)
# ex4_nrmlvols = ex4_vols['normals'].squeeze(-1)
# assert ex4_nrmlvols.shape == (n_bch, ex4_nvol, ex4_dim)
#
# integs = prob_ex4.integrate_volumes(ex4_vols)
#
# ex4_srfsamps = sphsampler_ex4(ex4_vols, ex4_nsrfpts, do_detspacing=True)
# ex4_points = ex4_srfsamps['points']
# assert ex4_points.shape == (n_bch, ex4_nvol, ex4_nsrfpts, ex4_dim)
# ex4_srfnrmls = ex4_srfsamps['normals']
# assert ex4_srfnrmls.shape == (n_bch, ex4_nvol, ex4_nsrfpts, ex4_dim)
# ex4_srftans = ex4_srfsamps['tangents']
# assert ex4_srftans.shape == (n_bch, ex4_nvol, ex4_nsrfpts, ex4_dim)

# %% tags=["active-ipynb"]
# # Sanity Check
# # Making sure that the disk normal vectors are prependicular 
# # to the point vectors
# aa = (ex4_points - ex4_cntrvols.unsqueeze(-2))
# assert aa.shape == (n_bch, ex4_nvol, ex4_nsrfpts, ex4_dim)
# bb = ex4_nrmlvols.reshape(n_bch, ex4_nvol, 1, ex4_dim)
#
# assert ((aa * bb).sum(dim=-1).abs() < 1e-8).all()

# %% tags=["active-ipynb"]
# ex4_fig = plt.figure(figsize=(4, 4), dpi=100)
# ex4_ax = ex4_fig.add_subplot(projection='3d')
# ex4_ax.set_aspect('equal')
# ex4_ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ex4_ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ex4_ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ex4_ax.set_xlim(-2., 2.)
# ex4_ax.set_ylim(-2., 2.)
# ex4_ax.set_zlim(-2., 2.)
#
# pts_plt = ex4_points.reshape(n_bch*ex4_nvol, ex4_nsrfpts, ex4_dim).detach().cpu().numpy()
# tngnts_plt = ex4_srftans.reshape(n_bch*ex4_nvol, ex4_nsrfpts, ex4_dim).detach().cpu().numpy()
# cntrs_plt = ex4_cntrvols.reshape(n_bch*ex4_nvol, ex4_dim).detach().cpu().numpy()
# dsknrmls_plt = ex4_nrmlvols.reshape(n_bch*ex4_nvol, ex4_dim).detach().cpu().numpy()
# for i_sph in [0, 2, 3]:
#     pts_iplt = pts_plt[i_sph]
#     tngs_iplt = tngnts_plt[i_sph] * 0.5
#     cntrs_iplt = cntrs_plt[i_sph]
#     dnrms_iplt = dsknrmls_plt[i_sph]
#     
#     ex4_ax.scatter(pts_iplt[:, 0], pts_iplt[:, 1], pts_iplt[:, 2], 
#         marker='o', c='rbbgcmyk'[i_sph], s=1)
#     
#     ex4_ax.quiver(pts_iplt[::10, 0], pts_iplt[::10, 1], pts_iplt[::10, 2], 
#         tngs_iplt[::10, 0], tngs_iplt[::10, 1], tngs_iplt[::10, 2], 
#         color='black')
#     
#     ex4_ax.quiver(cntrs_iplt[0], cntrs_iplt[1], cntrs_iplt[2], 
#         dnrms_iplt[0], dnrms_iplt[1], dnrms_iplt[2], 
#         color='orange')
#
# ex4_fig
# %% [markdown]
# ## Utility Functions for Sanity Checks

# %% code_folding=[8, 60, 85]
#########################################################
########### Sanity Checking Utility Functions ###########
#########################################################

msg_bcast = '{} should be np broadcastable to {}={}. '
msg_bcast += 'However, it has an inferred shape of {}.'


def get_arr(name, trgshp_str, trns_opts):
    """
    Gets a list of values, and checks if it is broadcastable to a 
    target shape. If the shape does not match, it will raise a proper
    assertion error with a meaninful message. The output is a numpy 
    array that is guaranteed to be broadcastable to the target shape.

    Parameters
    ----------
    name: (str) name of the option / hyper-parameter.

    trgshp_str: (str) the target shape elements representation. Must be a 
        valid python expression where the needed elements .

    trns_opts: (dict) a dictionary containing the variables needed 
        for the string to list translation of val.

    Key Variables
    -------------
    `val = trns_opts[name]`: (list or str) list of values read 
        from the config file. If a string is provided, python's 
        `eval` function will be used to translate it into a list.
        
    `trg_shape = eval_formula(trgshp_str, trns_opts)`: (tuple) 
        the target shape.
    
    Output
    ----------
    val_np: (np.array) the numpy array of val. 
    """
    msg_ =  f'"{name}" must be in trns_opts but it isnt: {trns_opts}'
    assert name in trns_opts, msg_
    val = trns_opts[name]
    
    if isinstance(val, str):
        val_list = eval_formula(val, trns_opts)
    else:
        val_list = val
    val_np = np.array(val_list)
    src_shape = val_np.shape
    trg_shape = eval_formula(trgshp_str, trns_opts)
    msg_ = msg_bcast.format(name, trgshp_str, trg_shape, src_shape)

    assert len(val_np.shape) == len(trg_shape), msg_

    is_bcastble = all((x == y or x == 1 or y == 1) for x, y in
                      zip(src_shape, trg_shape))
    assert is_bcastble, msg_

    return val_np


def eval_formula(formula, variables):
    """
    Gets a string formula and uses the `eval` function of python to  
    translate it into a python variable. The necessary variables for 
    translation are provided through the `variables` argument.

    Parameters
    ----------
    formula (str): a string that can be passed to `eval`.
        Example: "[np.sqrt(dim), 'a', None]"

    variables (dict): a dictionary of variables used in the formula.
        Example: {"dim": 4}

    Output
    ------
    pyobj (object): the translated formula into a python object
        Example: [2.0, 'a', None]

    """
    locals().update(variables)
    pyobj = eval(formula)
    return pyobj


def chck_dstrargs(opt, cfgdict, dstr2args, opt2req, parnt_optdstr=None):
    """
    Checks if the distribution arguments are provided correctly. Works 
    with hirarchical models through recursive applications. Proper error 
    messages are displayed if one of the checks fails.

    Parameters
    ----------
    opt: (str) the option name.

    cfgdict: (dict) the config dictionary.

    dstr2args: (dict) a mapping between distribution and their 
        required arguments.
        
    opt2req: (dict) required arguments for an option itself, not 
        necessarily required by the option's distribution.
    """
    opt_dstr = cfgdict.get(f'{opt}/dstr', 'fixed')

    msg_ = f'Unknown {opt}_dstr: it should be one of {list(dstr2args.keys())}'
    assert opt_dstr in dstr2args, msg_

    opt2req = dict() if opt2req is None else opt2req
    optreqs = opt2req.get(opt, tuple())
    must_spec = list(dstr2args[opt_dstr]) + list(optreqs)
    avid_spec = list(chain.from_iterable(
        v for k, v in dstr2args.items() if k != opt_dstr))
    avid_spec = [k for k in avid_spec if k not in must_spec]

    if opt_dstr == 'fixed':
        # To avoid infinite recursive calls, we should end this here.
        msg_ = f'"{opt}" must be specified.'
        if parnt_optdstr is not None:
            parnt_opt, parnt_dstr = parnt_optdstr
            msg_ += f'"{parnt_opt}" was specified as "{parnt_dstr}", and'
        msg_ += f' "{opt}" was specified as "{opt_dstr}".'
        if len(optreqs) > 0:
            msg_ += f' Also, "{opt}" requires "{optreqs}" to be specified.'
        opt_val = cfgdict.get(opt, None)
        assert opt_val is not None, msg_
    else:
        for arg in must_spec:
            opt_arg = f'{opt}{arg}'
            chck_dstrargs(opt_arg, cfgdict, dstr2args, opt2req, (opt, opt_dstr))

    for arg in avid_spec:
        opt_arg = f'{opt}{arg}'
        opt_arg_val = cfgdict.get(opt_arg, None)
        msg_ = f'"{opt_arg}" should not be specified, since "{opt}" '
        msg_ += f'appears to follow the "{opt_dstr}" distribution.'
        assert opt_arg_val is None, msg_



# %% [markdown]
# ## JSON Config Loading and Preprocessing

# %% code_folding=[1] tags=["active-ipynb"]
# json_cfgpath = f'../configs/03_maxwell/01_rect.json'
# ! rm -rf "./18_maxwell/results/01_rect.h5"
# ! rm -rf "./18_maxwell/storage/01_rect"
#
# if json_cfgpath.endswith('.json'):
#     with open(json_cfgpath, 'r') as fp:
#         json_cfgdict = json.load(fp, object_pairs_hook=odict)
# elif json_cfgpath.endswith('.yml'):
#     with open(json_cfgpath, "r") as fp:
#         json_cfgdict = odict(yaml.safe_load(fp))
# else:
#     raise RuntimeError(f'unknown config extension: {json_cfgpath}')
#
# json_cfgdict['io/config_id'] = '01_rect'
# json_cfgdict['io/results_dir'] = './18_maxwell/results'
# json_cfgdict['io/storage_dir'] = './18_maxwell/storage'
# json_cfgdict['io/tch/device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# all_cfgdicts = preproc_cfgdict(json_cfgdict)
# cfg_dict_input = all_cfgdicts[0]

# %% tags=["active-py"]
def main(cfg_dict_input):


# %% [markdown]
# ## Retrieving Config Variables

    # %%
    cfg_dict = cfg_dict_input.copy()

    #########################################################
    #################### Ignored Options ####################
    #########################################################
    cfgdesc = cfg_dict.pop('desc', None)
    cfgdate = cfg_dict.pop('date', None)

    #########################################################
    ################### Mandatory Options ###################
    #########################################################
    prob_type = cfg_dict.pop('problem')
    rng_seed_list = cfg_dict.pop('rng_seed/list')
    dim = cfg_dict.pop('dim')
    assert dim == 3

    n_srf = cfg_dict.pop('vol/n')
    n_srfpts_mdl = cfg_dict.pop('srfpts/n/mdl')
    n_srfpts_trg = cfg_dict.pop('srfpts/n/trg')
    do_detspacing = cfg_dict.pop('srfpts/detspc')
    do_dblsampling = cfg_dict.pop('srfpts/dblsmpl')

    do_bootstrap = cfg_dict.pop('trg/btstrp')
    if do_bootstrap:
        tau = cfg_dict.pop('trg/tau')
        w_trgreg = cfg_dict.pop('trg/reg/w')
    else:
        w_trgreg = 0.0
    w_trg = cfg_dict.pop('trg/w', None)

    opt_type = cfg_dict.pop('opt/dstr')
    n_epochs = cfg_dict.pop('opt/epoch')
    lr = cfg_dict.pop('opt/lr')

    #########################################################
    ################## Neural Spec Options ##################
    #########################################################
    nn_dstr = cfg_dict.pop('nn/dstr')
    if nn_dstr == 'mlp':
        nn_width = cfg_dict.pop('nn/width')
        nn_hidden = cfg_dict.pop('nn/hidden')
        nn_act = cfg_dict.pop('nn/act')
    else:
        msg_ = f'nn/dstr="{nn_dstr}" not defined!'
        raise ValueError(msg_)

    #########################################################
    ############## Charge Distribution Options ##############
    #########################################################
    wire_dstr = cfg_dict.pop('wire/dstr')
    wire_n = cfg_dict.pop('wire/n')

    wire_w_dstr = cfg_dict.pop('wire/w/dstr', 'fixed')
    if wire_w_dstr == 'fixed':
        wire_w_ = cfg_dict.pop('wire/w')
    elif wire_w_dstr == 'uniform':
        wire_w_low_ = cfg_dict.pop('wire/w/low', None)
        wire_w_high_ = cfg_dict.pop('wire/w/high', None)
    elif wire_w_dstr == 'normal':
        wire_w_loc_ = cfg_dict.pop('wire/w/loc', None)
        wire_w_scale_ = cfg_dict.pop('wire/w/scale', None)
    else:
        msg_ = f'wire/w/dstr="{wire_w_dstr}" not defined!'
        raise ValueError(msg_)

    wire_src_dstr = cfg_dict.pop('wire/src/dstr', 'fixed')
    if wire_src_dstr == 'fixed':
        wire_src_ = cfg_dict.pop('wire/src')
    elif wire_src_dstr == 'uniform':
        wire_src_low_ = cfg_dict.pop('wire/src/low')
        wire_src_high_ = cfg_dict.pop('wire/src/high')
    elif wire_src_dstr == 'normal':
        wire_src_loc_ = cfg_dict.pop('wire/src/loc')
        wire_src_scale_ = cfg_dict.pop('wire/src/scale')
    elif wire_src_dstr == 'ball':
        wire_src_c_ = cfg_dict.pop('wire/src/c')
        wire_src_r_ = cfg_dict.pop('wire/src/r')
    else:
        msg_ = f'wire/src/dstr="{wire_src_dstr}" not defined!'
        raise ValueError(msg_)
    
    wire_snk_dstr = cfg_dict.pop('wire/snk/dstr', 'fixed')
    if wire_snk_dstr == 'fixed':
        wire_snk_ = cfg_dict.pop('wire/snk')
    elif wire_snk_dstr == 'uniform':
        wire_snk_low_ = cfg_dict.pop('wire/snk/low')
        wire_snk_high_ = cfg_dict.pop('wire/snk/high')
    elif wire_snk_dstr == 'normal':
        wire_snk_loc_ = cfg_dict.pop('wire/snk/loc')
        wire_snk_scale_ = cfg_dict.pop('wire/snk/scale')
    elif wire_snk_dstr == 'ball':
        wire_snk_c_ = cfg_dict.pop('wire/snk/c')
        wire_snk_r_ = cfg_dict.pop('wire/snk/r')
    else:
        msg_ = f'wire/snk/dstr="{wire_snk_dstr}" not defined!'
        raise ValueError(msg_)
    
    #########################################################
    ############ Problem Parameter Search Options ###########
    #########################################################
    srch_dstr = cfg_dict.pop('srch/dstr', None)
    if srch_dstr in ('mcmc',):
        srch_enc = cfg_dict.pop('srch/enc')
        srch_prior_dstr = cfg_dict.pop('srch/prior/dstr')
        assert srch_prior_dstr == 'normal'
        srch_prior_loc_ = cfg_dict.pop('srch/prior/loc')
        srch_prior_scale_ = cfg_dict.pop('srch/prior/scale')
    elif srch_dstr is None:
        pass
    else:
        msg_ = f'srch/dstr="{srch_dstr}" not defined!'
        raise ValueError(msg_)
    
    # Shared parameters for all search methods
    if srch_dstr is not None:
        obs_x_dstr = cfg_dict.pop('srch/obs/x/dstr')
        assert obs_x_dstr == 'uniform'
        obs_x_low_ = cfg_dict.pop('srch/obs/x/low')
        obs_x_high_ = cfg_dict.pop('srch/obs/x/high')
        
        obs_y_dstr = cfg_dict.pop('srch/obs/y/dstr')
        obs_n = cfg_dict.pop('srch/obs/n')
        
        srch_metric_dstr = cfg_dict.pop('srch/metric/dstr')
        srch_metric_coeff = cfg_dict.pop('srch/metric/coeff')
        srch_metric_min = cfg_dict.pop('srch/metric/min', None)
        srch_metric_max = cfg_dict.pop('srch/metric/max', None)
        
        srch_frq = cfg_dict.pop('srch/frq')
        srch_inittrn = cfg_dict.pop('srch/inittrn')
        srch_tmprtr = cfg_dict.pop('srch/tmprtr')
        srch_sigma = cfg_dict.pop('srch/sigma')
        n_grps = cfg_dict.pop('srch/grps')
        srch_reset_n = cfg_dict.pop('srch/reset/n')
        srch_reset_k = cfg_dict.pop('srch/reset/k')
    
    #########################################################
    ############### Surface Sampling Options ################
    #########################################################
    vol_dstr = cfg_dict.pop('vol/dstr')

    vol_c_dstr = cfg_dict.pop('vol/c/dstr', 'fixed')
    if vol_c_dstr == 'fixed':
        vol_c_ = cfg_dict.pop('vol/c')
    elif vol_c_dstr == 'uniform':
        vol_c_low_ = cfg_dict.pop('vol/c/low')
        vol_c_high_ = cfg_dict.pop('vol/c/high')
    elif vol_c_dstr == 'normal':
        vol_c_loc_ = cfg_dict.pop('vol/c/loc')
        vol_c_scale_ = cfg_dict.pop('vol/c/scale')
    elif vol_c_dstr == 'ball':
        vol_c_c_ = cfg_dict.pop('vol/c/c')
        vol_c_r_ = cfg_dict.pop('vol/c/r')
    else:
        msg_ = f'vol/c/dstr="{vol_c_dstr}" not defined!'
        raise ValueError(msg_)

    vol_r_dstr = cfg_dict.pop('vol/r/dstr', 'fixed')
    if vol_r_dstr == 'fixed':
        vol_r_ = cfg_dict.pop('vol/r')
    elif vol_r_dstr in ('uniform', 'unifdpow'):
        vol_r_low_ = cfg_dict.pop('vol/r/low')
        vol_r_high_ = cfg_dict.pop('vol/r/high')
    else:
        msg_ = f'vol/r/dstr="{vol_r_dstr}" not defined!'
        raise ValueError(msg_)
    
    vol_nv_dstr = cfg_dict.pop('vol/nv/dstr', 'fixed')
    if vol_nv_dstr == 'fixed':
        vol_nv_ = cfg_dict.pop('vol/nv')
    elif vol_nv_dstr in ('uball',):
        pass
    else:
        msg_ = f'vol/nv/dstr="{vol_nv_dstr}" not defined!'
        raise ValueError(msg_)

    #########################################################
    ############## Initial Condition Options ###############
    #########################################################
    ic_dstr = cfg_dict.pop('ic/dstr', None)
    if ic_dstr in ('sphere', 'trnvol'):
        w_ic = cfg_dict.pop('ic/w')
        ic_bpp = cfg_dict.pop('ic/bpp')
        ic_n = cfg_dict.pop('ic/n')
        ic_frq = cfg_dict.pop('ic/frq')
        ic_bs = cfg_dict.pop('ic/bs')
        ic_needsampling = True
    elif ic_dstr in ('trnsrf',):
        w_ic = cfg_dict.pop('ic/w')
        ic_bpp = cfg_dict.pop('ic/bpp')
        ic_n = n_srf * n_srfpts_mdl
        ic_frq = cfg_dict.pop('ic/frq')
        ic_bs = ic_n
        ic_needsampling = False
    elif ic_dstr is None:
        ic_needsampling = False
        ic_frq = 1
        w_ic, ic_bpp = 0, 'all'
    else:
        msg_ = f'ic/dstr={ic_dstr} not defined'
        raise ValueError(msg_)

    if ic_dstr == 'sphere':
        ic_c_dstr = cfg_dict.pop('ic/c/dstr', 'fixed')
        if ic_c_dstr == 'fixed':
            ic_c_ = cfg_dict.pop('ic/c')
        elif ic_c_dstr == 'uniform':
            ic_c_low_ = cfg_dict.pop('ic/c/low')
            ic_c_high_ = cfg_dict.pop('ic/c/high')
        elif ic_c_dstr == 'normal':
            ic_c_loc_ = cfg_dict.pop('ic/c/loc')
            ic_c_scale_ = cfg_dict.pop('ic/c/scale')
        else:
            msg_ = f'ic/c/dstr="{ic_c_dstr}" not defined!'
            raise ValueError(msg_)

        ic_r_dstr = cfg_dict.pop('ic/r/dstr', 'fixed')
        if ic_r_dstr == 'fixed':
            ic_r_ = cfg_dict.pop('ic/r')
        elif ic_r_dstr in ('uniform', 'unifdpow'):
            ic_r_low_ = cfg_dict.pop('ic/r/low')
            ic_r_high_ = cfg_dict.pop('ic/r/high')
        else:
            msg_ = f'ic/r/dstr="{ic_r_dstr}" not defined!'
            raise ValueError(msg_)
    elif ic_dstr in ('trnsrf', 'trnvol', None):
        pass
    else:
        msg_ = f'ic/dstr="{ic_dstr}" not defined!'
        raise ValueError(msg_)

    #########################################################
    ########### Evaluation Point Sampling Options ###########
    #########################################################
    eid_list_dup = [opt.split('/')[1] for opt in cfg_dict
                    if opt.startswith('eval/')]
    eid_list = list(odict.fromkeys(eid_list_dup))
    evalcfgs = odict()
    for eid in eid_list:
        evalcfgs[eid] = odict()
        cfgopts = list(cfg_dict.keys())
        for opt in cfgopts:
            prfx = f'eval/{eid}/'
            if opt.startswith(prfx):
                optn = opt[len(prfx):]
                optv = cfg_dict.pop(opt)
                evalcfgs[eid][optn] = optv

    #########################################################
    ################# I/O Logistics Options #################
    #########################################################
    config_id = cfg_dict.pop('io/config_id')
    results_dir = cfg_dict.pop('io/results_dir')
    storage_dir = cfg_dict.pop('io/storage_dir', None)
    io_avgfrq = cfg_dict.pop('io/avg/frq')
    ioflsh_period = cfg_dict.pop('io/flush/frq')
    chkpnt_period = cfg_dict.pop('io/ckpt/frq')
    device_name = cfg_dict.pop('io/tch/device')
    dtype_name = cfg_dict.pop('io/tch/dtype')
    iomon_period = cfg_dict.pop('io/mon/frq')
    io_cmprssnlvl = cfg_dict.pop('io/cmprssn_lvl')
    eval_bs = cfg_dict.pop('io/eval/bs', None)

    dtnow = datetime.datetime.now().isoformat(timespec='seconds')
    hostname = socket.gethostname()
    commit_hash = get_git_commit()
    cfg_tree = '/'.join(config_id.split('/')[:-1])
    cfg_name = config_id.split('/')[-1]
    #########################################################
    ##################### Sanity Checks #####################
    #########################################################

    # Making sure the specified option distributions are implemented.
    fixed_opts = ['desc', 'date', 'rng_seed/list', 'problem',
                  'dim', 'vol/n',  'srfpts/n/mdl', 'srfpts/n/trg', 
                  'srfpts/detspc', 'srfpts/dblsmpl','trg/btstrp', 
                  'trg/w', 'trg/tau', 'opt/lr', 'opt/epoch']

    opt2availdstrs = {**{opt: ('fixed',) for opt in fixed_opts},
        'wire': ('dmm',), 'wire/n': ('fixed',), 'wire/w': ('fixed',), 
        'wire/src': ('fixed', 'uniform', 'normal', 'ball'),
        'wire/snk': ('fixed', 'uniform', 'normal', 'ball'),
        'vol': ('disk',), 'vol/c': ('uniform', 'ball', 'normal'), 
        'vol/r': ('uniform', 'unifdpow'), 'vol/nv': ('fixed', 'uball'), 
        'ic': ('sphere', 'trnsrf', 'trnvol', 'fixed'), 
        'ic/c': ('fixed',), 'nn': ('mlp',), 'ic/r': ('fixed',), 
        'vol/c/low': ('fixed',), 'vol/c/high': ('fixed',),
        'vol/c/loc': ('fixed',), 'vol/c/scale': ('fixed',),
        'vol/c/c': ('fixed',),   'vol/c/r': ('fixed',),
        'vol/r/low': ('fixed',), 'vol/r/high': ('fixed',),
        **{f'eval/{eid}': ('uniform', 'grid', 'ball', 'trnvol')
            for eid in eid_list}}

    for opt, avail_dstrs in opt2availdstrs.items():
        opt_dstr = cfg_dict_input.get(f'{opt}/dstr', 'fixed')
        msg_  = f'"{opt}" cannot follow the "{opt_dstr}" distribution or type '
        msg_ += f' since it is not implemented or available at least yet. The '
        msg_ += f'only available options for "{opt}" are {avail_dstrs}.'
        assert opt_dstr in avail_dstrs, msg_

    # Making sure no other options are left unused.
    if len(cfg_dict) > 0:
        msg_ = f'The following settings were left unused:\n'
        for key, val in cfg_dict.items():
            msg_ += f'  {key}: {val}'
        raise RuntimeError(msg_)

    # Making sure that all "*_dstr" options are valid
    dstr2args = {'uniform':  ('/low', '/high'),
                 'unifdpow': ('/low', '/high'),
                 'normal':   ('/loc', '/scale'),
                 'dmm':      ('/n', '/w', '/src', '/snk'),
                 'disk':     ('/c', '/r', '/nv'),
                 'ball':     ('/c', '/r'),
                 'sphere':   ('/c', '/r'),
                 'fixed':    ('',),
                 'uball':    (),
                 'trnvol':   (),
                 'trnsrf':   ()}

    key2req = {'vol': ('/n',)}
    if ic_dstr is not None:
        key2req['ic'] = (*key2req['ic'], '/w')
    if ic_needsampling: 
        key2req['ic'] = (*key2req['ic'], '/n', '/frq')

    for key in ['wire', 'vol']:
        if key in key2req:
            chck_dstrargs(key, cfg_dict_input, dstr2args, key2req)

    edstr2args = {'uniform': ('/low', '/high', '/n', '/frq'),
                  'grid':    ('/low', '/high', '/n', '/frq'),
                  'ball':    ('/c', '/r', '/n', '/frq'),
                  'fixed':   ('', '/n', '/frq'),
                  'trnvol':  ('/n', '/frq')}
    for eid in eid_list:
        chck_dstrargs(f'eval/{eid}', cfg_dict_input, 
            edstr2args, None)


# %% [markdown]
# ## Problem Objects Construction 

    # %%
    # Derived options and assertions
    n_points = n_srfpts_mdl + n_srfpts_trg

    assert not (do_dblsampling) or (n_srfpts_trg > 1)
    if w_trg is None:
        w_trg = n_srfpts_trg / n_points
    assert not (n_srfpts_mdl == 0) or (w_trg == 1.0)
    n_rsdls = 2 if do_dblsampling else 1

    if eval_bs is None:
        eval_bs = max(n_srfpts_mdl, n_srfpts_trg) * n_srf
        
    #########################################################
    ########### I/O-Related Options and Operations ##########
    #########################################################

    name2dtype = dict(float64=torch.double,
                      float32=torch.float32,
                      float16=torch.float16)
    tch_device = torch.device(device_name)
    tch_dtype = name2dtype[dtype_name]

    tch_dvcmdl = device_name
    if device_name.startswith('cuda'):
        tch_dvcmdl = torch.cuda.get_device_name(tch_device)

    # Reserving 15.596 GB of memory for later usage
    # t_gpumem = torch.cuda.get_device_properties(tch_device).total_memory
    # tdt_elsize = torch.tensor(1).to(tch_device, tch_dtype).element_size()
    # nuslss = int((0.90 * t_gpumem) / tdt_elsize)
    # useless_tensor = torch.empty((nuslss,), device=tch_device, dtype=tch_dtype)
    # del useless_tensor
    
    msg_ = f'"io/mon/frq" % "io/avg/frq" != 0'
    assert iomon_period % io_avgfrq == 0, msg_
    msg_ = f'"io/ckpt/frq" % "io/avg/frq" != 0'
    assert chkpnt_period % io_avgfrq == 0, msg_
    
    do_logtb = storage_dir is not None
    do_profile = False
    do_tchsave = storage_dir is not None
    
    assert not(do_logtb) or (storage_dir is not None)
    assert not(do_profile) or (storage_dir is not None)
    assert not(do_tchsave) or (storage_dir is not None)

    #########################################################
    ########### Constructing the Batch RNG Object ###########
    #########################################################
    n_seeds = len(rng_seed_list)
    rng_seeds = np.array(rng_seed_list)
    rng = BatchRNG(shape=(n_seeds,), lib='torch',
                   device=tch_device, dtype=tch_dtype,
                   unif_cache_cols=1_000_000,
                   norm_cache_cols=5_000_000)
    rng.seed(np.broadcast_to(rng_seeds, rng.shape))

    #########################################################
    ########## Defining the Maxwell Problem Object ##########
    #########################################################
    assert prob_type == 'maxwell'

    msg_ = f'wire_dstr = {wire_dstr} is not available/implemented.'
    assert wire_dstr in ('dmm',), msg_

    trns_opts = dict(dim=dim, wire_n=wire_n, sqrt=np.sqrt)

    # The wire currents
    if wire_w_dstr == 'fixed':
        wire_w_0 = get_arr('wire_w', '(wire_n,)', 
            {**trns_opts, 'wire_w': wire_w_})
        wire_w = np.broadcast_to(wire_w_0[None, ...],
                                 (n_seeds, wire_n)).copy()
        assert wire_w.shape == (n_seeds, wire_n)
    else:
        raise ValueError(f'wire_w_dstr={wire_w_dstr} '
                         'not implemented.')

    # The Maxwell wire current source points
    if wire_src_dstr == 'fixed':
        wire_src_0 = get_arr('wire_src', '(wire_n, dim)', 
            {**trns_opts, 'wire_src': wire_src_})
        wire_src = np.broadcast_to(wire_src_0[None, ...],
                                  (n_seeds, wire_n, dim)).copy()
        assert wire_src.shape == (n_seeds, wire_n, dim)
    elif wire_src_dstr == 'uniform':
        wire_src_low_0 =  get_arr('wire_src_low', '(wire_n, dim)', 
            {**trns_opts, 'wire_src_low': wire_src_low_})
        wire_src_low = np.broadcast_to(wire_src_low_0[None, ...],
            (n_seeds, wire_n, dim)).copy()
        assert wire_src_low.shape == (n_seeds, wire_n, dim)
        
        wire_src_high_0 =  get_arr('wire_src_high', '(wire_n, dim)', 
            {**trns_opts, 'wire_src_high': wire_src_high_})
        wire_src_high = np.broadcast_to(wire_src_high_0[None, ...],
            (n_seeds, wire_n, dim)).copy()
        assert wire_src_high.shape == (n_seeds, wire_n, dim)
        
        rnds = rng.uniform((n_seeds, wire_n, dim)).detach().cpu().numpy()
        wire_src = wire_src_low + rnds * (wire_src_high - wire_src_low)
        assert wire_src.shape == (n_seeds, wire_n, dim)
    elif wire_src_dstr == 'normal':
        wire_src_loc_0 =  get_arr('wire_src_loc', '(wire_n, dim)', 
            {**trns_opts, 'wire_src_loc': wire_src_loc_})
        wire_src_loc = np.broadcast_to(wire_src_loc_0[None, ...],
            (n_seeds, wire_n, dim)).copy()
        assert wire_src_loc.shape == (n_seeds, wire_n, dim)
        
        wire_src_scale_0 =  get_arr('wire_src_scale', '(wire_n,)', 
            {**trns_opts, 'wire_src_scale': wire_src_scale_})
        wire_src_scale = np.broadcast_to(wire_src_scale_0[None, ...],
            (n_seeds, wire_n)).copy()
        assert wire_src_scale.shape == (n_seeds, wire_n)
        
        rnds = rng.normal((n_seeds, wire_n, dim)).detach().cpu().numpy()
        wire_src = wire_src_loc + rnds * wire_src_scale.reshape(n_seeds, wire_n, 1)
        assert wire_src.shape == (n_seeds, wire_n, dim)
    elif wire_src_dstr == 'ball':
        wire_src_c_0 =  get_arr('wire_src_c', '(wire_n, dim)', 
            {**trns_opts, 'wire_src_c': wire_src_c_})
        wire_src_c = np.broadcast_to(wire_src_c_0[None, ...],
            (n_seeds, wire_n, dim)).copy()
        assert wire_src_c.shape == (n_seeds, wire_n, dim)
        
        wire_src_r_0 =  get_arr('wire_src_r', '(wire_n,)', 
            {**trns_opts, 'wire_src_r': wire_src_r_})
        wire_src_r = np.broadcast_to(wire_src_r_0[None, ...],
            (n_seeds, wire_n)).copy()
        assert wire_src_r.shape == (n_seeds, wire_n)
        
        rnds1 = rng.normal((n_seeds, wire_n, dim)).detach().cpu().numpy()
        rnds1_tilde = rnds1 / np.sqrt((rnds1**2).sum(axis=-1, keepdims=True))
        assert rnds1_tilde.shape == (n_seeds, wire_n, dim)
        
        rnds2 = rng.uniform((n_seeds, wire_n)).detach().cpu().numpy()
        rnds2_tilde = rnds2 ** (1.0 / dim)
        assert rnds2_tilde.shape == (n_seeds, wire_n)
        
        rnds3_tilde = (wire_src_r * rnds2_tilde).reshape(n_seeds, wire_n, 1)
        assert rnds3_tilde.shape == (n_seeds, wire_n, 1)
        
        wire_src = wire_src_c + rnds1_tilde * rnds3_tilde
        assert wire_src.shape == (n_seeds, wire_n, dim)
    else:
        raise ValueError(f'wire_src_dstr={wire_src_dstr} '
                         'not implemented.')

    # The Maxwell wire current sink points
    # Note: The following is a copy of the above current source definition
    #       with the `_src` prefixes being replaced with `_snk`  
    if wire_snk_dstr == 'fixed':
        wire_snk_0 = get_arr('wire_snk', '(wire_n, dim)', 
            {**trns_opts, 'wire_snk': wire_snk_})
        wire_snk = np.broadcast_to(wire_snk_0[None, ...],
                                  (n_seeds, wire_n, dim)).copy()
        assert wire_snk.shape == (n_seeds, wire_n, dim)
    elif wire_snk_dstr == 'uniform':
        wire_snk_low_0 =  get_arr('wire_snk_low', '(wire_n, dim)', 
            {**trns_opts, 'wire_snk_low': wire_snk_low_})
        wire_snk_low = np.broadcast_to(wire_snk_low_0[None, ...],
            (n_seeds, wire_n, dim)).copy()
        assert wire_snk_low.shape == (n_seeds, wire_n, dim)
        
        wire_snk_high_0 =  get_arr('wire_snk_high', '(wire_n, dim)', 
            {**trns_opts, 'wire_snk_high': wire_snk_high_})
        wire_snk_high = np.broadcast_to(wire_snk_high_0[None, ...],
            (n_seeds, wire_n, dim)).copy()
        assert wire_snk_high.shape == (n_seeds, wire_n, dim)
        
        rnds = rng.uniform((n_seeds, wire_n, dim)).detach().cpu().numpy()
        wire_snk = wire_snk_low + rnds * (wire_snk_high - wire_snk_low)
        assert wire_snk.shape == (n_seeds, wire_n, dim)
    elif wire_snk_dstr == 'normal':
        wire_snk_loc_0 =  get_arr('wire_snk_loc', '(wire_n, dim)', 
            {**trns_opts, 'wire_snk_loc': wire_snk_loc_})
        wire_snk_loc = np.broadcast_to(wire_snk_loc_0[None, ...],
            (n_seeds, wire_n, dim)).copy()
        assert wire_snk_loc.shape == (n_seeds, wire_n, dim)
        
        wire_snk_scale_0 =  get_arr('wire_snk_scale', '(wire_n,)', 
            {**trns_opts, 'wire_snk_scale': wire_snk_scale_})
        wire_snk_scale = np.broadcast_to(wire_snk_scale_0[None, ...],
            (n_seeds, wire_n)).copy()
        assert wire_snk_scale.shape == (n_seeds, wire_n)
        
        rnds = rng.normal((n_seeds, wire_n, dim)).detach().cpu().numpy()
        wire_snk = wire_snk_loc + rnds * wire_snk_scale.reshape(n_seeds, wire_n, 1)
        assert wire_snk.shape == (n_seeds, wire_n, dim)
    elif wire_snk_dstr == 'ball':
        wire_snk_c_0 =  get_arr('wire_snk_c', '(wire_n, dim)', 
            {**trns_opts, 'wire_snk_c': wire_snk_c_})
        wire_snk_c = np.broadcast_to(wire_snk_c_0[None, ...],
            (n_seeds, wire_n, dim)).copy()
        assert wire_snk_c.shape == (n_seeds, wire_n, dim)
        
        wire_snk_r_0 =  get_arr('wire_snk_r', '(wire_n,)', 
            {**trns_opts, 'wire_snk_r': wire_snk_r_})
        wire_snk_r = np.broadcast_to(wire_snk_r_0[None, ...],
            (n_seeds, wire_n)).copy()
        assert wire_snk_r.shape == (n_seeds, wire_n)
        
        rnds4 = rng.normal((n_seeds, wire_n, dim)).detach().cpu().numpy()
        rnds4_tilde = rnds4 / np.sqrt((rnds4**2).sum(axis=-1, keepdims=True))
        assert rnds4_tilde.shape == (n_seeds, wire_n, dim)
        
        rnds5 = rng.uniform((n_seeds, wire_n)).detach().cpu().numpy()
        rnds5_tilde = rnds5 ** (1.0 / dim)
        assert rnds5_tilde.shape == (n_seeds, wire_n)
        
        rnds6_tilde = (wire_snk_r * rnds5_tilde).reshape(n_seeds, wire_n, 1)
        assert rnds6_tilde.shape == (n_seeds, wire_n, 1)
        
        wire_snk = wire_snk_c + rnds1_tilde * rnds6_tilde
        assert wire_snk.shape == (n_seeds, wire_n, dim)
    else:
        raise ValueError(f'wire_src_dstr={wire_src_dstr} '
                         'not implemented.')
        

    # Defining the problem object
    problem = DeltaLineProblem(source=wire_src, sink=wire_snk, current=wire_w,
        tch_device=tch_device, tch_dtype=tch_dtype)
    
    #########################################################
    ######### Defining the Latent Parameter Search ##########
    #########################################################
    if srch_dstr in ('mcmc',):
        true_problem = problem
        
        mcmc_enc = MCMCWireEncoder(kind=srch_enc, source=wire_src, sink=wire_snk, current=wire_w)
        assert srch_prior_dstr == 'normal'
        
        # The search dimension
        sdim = mcmc_enc.sdim
        trns_opts['sdim'] = sdim
        
        srch_prior_loc_0 =  get_arr('srch_prior_loc', '(sdim,)', 
            {**trns_opts, 'srch_prior_loc': srch_prior_loc_})
        srch_prior_loc = np.broadcast_to(srch_prior_loc_0[None, ...],
            (n_seeds, sdim)).copy()
        assert srch_prior_loc.shape == (n_seeds, sdim)
        
        srch_prior_scale_0 =  get_arr('srch_prior_scale', '(sdim,)', 
            {**trns_opts, 'srch_prior_scale': srch_prior_scale_})
        srch_prior_scale = np.broadcast_to(srch_prior_scale_0[None, ...],
            (n_seeds, sdim)).copy()
        assert srch_prior_scale.shape == (n_seeds, sdim)
        
        srch_prior_loc_tch = torch.from_numpy(srch_prior_loc)
        srch_prior_loc_tch = srch_prior_loc_tch.to(device=tch_device, dtype=tch_dtype)
        assert srch_prior_loc_tch.shape == (n_seeds, sdim)
        
        srch_prior_scale_tch = torch.from_numpy(srch_prior_scale)
        srch_prior_scale_tch = srch_prior_scale_tch.to(device=tch_device, dtype=tch_dtype)
        assert srch_prior_scale_tch.shape == (n_seeds, sdim)
        
        rnds = rng.normal((n_seeds, sdim))
        old_srch_mu = srch_prior_loc_tch + rnds * srch_prior_scale_tch
        assert old_srch_mu.shape == (n_seeds, sdim)
        
        old_srch_mu_np = old_srch_mu.detach().cpu().numpy()
        assert old_srch_mu_np.shape == (n_seeds, sdim)
        
        old_srch_params = mcmc_enc(old_srch_mu_np)
        
        # Re-creating the problem instance with randomized initial parameters for search
        problem = DeltaLineProblem(source=old_srch_params['source'], 
            sink=old_srch_params['sink'], current=old_srch_params['current'],
            tch_device=tch_device, tch_dtype=tch_dtype)

        old_logpri, old_loglike, old_logpost = None, None, None
        
        # Total number of MCMC search draws
        n_srchdraws = (n_epochs - srch_inittrn) // srch_frq
    elif srch_dstr is None:
        pass
    else:
        msg_ = f'srch/dstr="{srch_dstr}" not defined!'
        raise ValueError(msg_)
    
    if srch_dstr is not None:
        n_cpg = n_seeds // n_grps
        assert n_seeds == (n_grps * n_cpg)
        
        assert n_cpg % srch_reset_k == 0
        
        assert obs_x_dstr == 'uniform'
        
        obs_x_low_0 =  get_arr('obs_x_low', '(obs_n, dim)', 
            {**trns_opts, 'obs_x_low': obs_x_low_})
        obs_x_low = np.broadcast_to(obs_x_low_0[None, ...],
            (n_seeds, obs_n, dim)).copy()
        assert obs_x_low.shape == (n_seeds, obs_n, dim)
        
        obs_x_high_0 =  get_arr('obs_x_high', '(obs_n, dim)', 
            {**trns_opts, 'obs_x_high': obs_x_high_})
        obs_x_high = np.broadcast_to(obs_x_high_0[None, ...],
            (n_seeds, obs_n, dim)).copy()
        assert obs_x_high.shape == (n_seeds, obs_n, dim)
        
        rnds = rng.uniform((n_seeds, obs_n, dim))
        
        # Creating the x observations, and making sure that each 
        # group gets the same set of observations.
        obs_x_low_tch = torch.from_numpy(obs_x_low).to(device=tch_device, dtype=tch_dtype)
        obs_x_high_tch = torch.from_numpy(obs_x_high).to(device=tch_device, dtype=tch_dtype)
        
        obs_x1 = obs_x_low_tch + rnds * (obs_x_high_tch - obs_x_low_tch)
        assert obs_x1.shape == (n_seeds, obs_n, dim)
        obs_x2 = obs_x1.reshape(n_grps, n_cpg, obs_n, dim)
        assert obs_x2.shape == (n_grps, n_cpg, obs_n, dim)
        obs_x3 = obs_x2[:, :1, ...]
        assert obs_x3.shape == (n_grps, 1, obs_n, dim)
        obs_x4 = obs_x3.expand(n_grps, n_cpg, obs_n, dim)
        assert obs_x4.shape == (n_grps, n_cpg, obs_n, dim)
        obs_x = obs_x4.reshape(n_seeds, obs_n, dim)
        assert obs_x.shape == (n_seeds, obs_n, dim)
        
        if obs_y_dstr == 'potential':
            obs_ydim = dim
            obs_y = true_problem.potential(obs_x)
            assert obs_y.shape == (n_seeds, obs_n, obs_ydim)
            obs_y = obs_y - obs_y.mean(dim=-2, keepdim=True)
            assert obs_y.shape == (n_seeds, obs_n, obs_ydim)
        elif obs_y_dstr == 'field':
            obs_ydim = dim
            obs_y = true_problem.field(obs_x)
            assert obs_y.shape == (n_seeds, obs_n, obs_ydim)
        else:
            raise ValueError(f'obs_y_dstr={obs_y_dstr} not implemented.')

    #########################################################
    ####### Defining the Initial Condition Parameters #######
    #########################################################
    msg_ = f'ic/dstr = {ic_dstr} is not available/implemented.'
    assert ic_dstr in ('sphere', 'trnsrf', None), msg_
    msg_ = '"ic/bpp" must be either "bias" or "all".'
    assert ic_bpp in ('bias', 'all', None), msg_

    if ic_dstr == 'sphere':
        if ic_c_dstr == 'fixed':
            ic_c_0_np = get_arr('ic_c', '(dim,)', 
                {**trns_opts, 'ic_c': ic_c_})
            ic_c_np = np.broadcast_to(ic_c_0_np[None, ...], 
                (n_seeds, dim)).copy()
            assert ic_c_np.shape == (n_seeds, dim)
        else:
            raise ValueError(f'ic_c_dstr={ic_c_dstr} '
                            'not implemented.')

        if ic_r_dstr == 'fixed':
            ic_r_0_np = get_arr('ic_r', '()', 
                {**trns_opts, 'ic_r': ic_r_})
            ic_r_np = np.broadcast_to(ic_r_0_np[None, ...], 
                (n_seeds,)).copy()
            assert ic_r_np.shape == (n_seeds,)
        else:
            raise ValueError(f'ic_r_dstr={ic_r_dstr} '
                            'not implemented.')

        with torch.no_grad():
            ic_c = torch.from_numpy(ic_c_np).to(device=tch_device, 
                dtype=tch_dtype).reshape(n_seeds, 1, dim).expand(n_seeds, ic_n, dim)
            assert ic_c.shape == (n_seeds, ic_n, dim)

            ic_r = torch.from_numpy(ic_r_np).to(device=tch_device, 
                dtype=tch_dtype).reshape(n_seeds, 1, 1).expand(n_seeds, ic_n, 1)
            assert ic_r.shape == (n_seeds, ic_n, 1)
    elif ic_dstr in ('trnsrf', 'trnvol', None):
        pass
    else:
        raise ValueError(f'ic_dstr={ic_dstr} not implemented')

    #########################################################
    ########## Defining the Volume Sampling Object ##########
    #########################################################
    msg_ = f'vol/dstr = {vol_dstr} is not available/implemented.'
    assert vol_dstr in ('disk',), msg_

    if vol_c_dstr == 'uniform':
        vol_c_low_0 = get_arr('vol_c_low', '(dim,)', 
            {**trns_opts, 'vol_c_low': vol_c_low_})
        vol_c_low = np.broadcast_to(vol_c_low_0[None, ...],
            (n_seeds, dim)).copy()
        assert vol_c_low.shape == (n_seeds, dim)

        vol_c_high_0 = get_arr('vol_c_high', '(dim,)', 
            {**trns_opts, 'vol_c_high': vol_c_high_})
        vol_c_high = np.broadcast_to(vol_c_high_0[None, ...],
            (n_seeds, dim)).copy()
        assert vol_c_high.shape == (n_seeds, dim)
        
        vol_c_params = dict(low=vol_c_low, high=vol_c_high)
    elif vol_c_dstr == 'normal':
        vol_c_loc_0 = get_arr('vol_c_loc', '(dim,)', 
            {**trns_opts, 'vol_c_loc': vol_c_loc_})
        vol_c_loc = np.broadcast_to(vol_c_loc_0[None, ...],
            (n_seeds, dim)).copy()
        assert vol_c_loc.shape == (n_seeds, dim)

        vol_c_scale_0 = get_arr('vol_c_scale', '()', 
            {**trns_opts, 'vol_c_scale': vol_c_scale_})
        vol_c_scale = np.broadcast_to(vol_c_scale_0[None, ...],
            (n_seeds,)).copy()
        assert vol_c_scale.shape == (n_seeds,)
        
        vol_c_params = dict(loc=vol_c_loc, scale=vol_c_scale)
    elif vol_c_dstr == 'ball':
        vol_c_c_0 = get_arr('vol_c_c', '(dim,)', 
            {**trns_opts, 'vol_c_c': vol_c_c_})
        vol_c_c = np.broadcast_to(vol_c_c_0[None, ...],
            (n_seeds, dim)).copy()
        assert vol_c_c.shape == (n_seeds, dim)

        vol_c_r_0 = get_arr('vol_c_r', '()', 
            {**trns_opts, 'vol_c_r': vol_c_r_})
        vol_c_r = np.broadcast_to(vol_c_r_0[None, ...],
            (n_seeds,)).copy()
        assert vol_c_r.shape == (n_seeds,)
        
        vol_c_params = dict(c=vol_c_c, r=vol_c_r)
    else:
        raise ValueError(f'vol_c_dstr={vol_c_dstr} not implemented.')

    if vol_r_dstr in ('uniform', 'unifdpow'):
        vol_r_low_0 = get_arr('vol_r_low', '()', 
            {**trns_opts, 'vol_r_low': vol_r_low_, 
             'vol_r_high': vol_r_high_})
        vol_r_low = np.broadcast_to(vol_r_low_0[None, ...],
            (n_seeds,)).copy()
        assert vol_r_low.shape == (n_seeds,)

        vol_r_high_0 = get_arr('vol_r_high', '()', 
            {**trns_opts, 'vol_r_low': vol_r_low_, 
             'vol_r_high': vol_r_high_})
        vol_r_high = np.broadcast_to(vol_r_high_0[None, ...],
            (n_seeds,)).copy()
        assert vol_r_high.shape == (n_seeds,)
        
        vol_r_params = dict(low=vol_r_low, high=vol_r_high)
    else:
        raise ValueError(f'vol_r_dstr={vol_r_dstr} not implemented.')
    
    if vol_nv_dstr == 'uball':
        vol_nv_params = dict()
    elif vol_nv_dstr == 'fixed':
        vol_nv_0 = get_arr('vol_nv', '(dim,)', 
            {**trns_opts, 'vol_nv': vol_nv_})
        vol_nv = np.broadcast_to(vol_nv_0[None, ...],
            (n_seeds, dim)).copy()
        assert vol_nv.shape == (n_seeds, dim)
        
        vol_nv_params = dict(value=vol_nv)
    else:
        raise ValueError(f'vol_nv_dstr={vol_nv_dstr} not implemented.')
        

    volsampler = DiskSampler(c_dstr=vol_c_dstr, c_params=vol_c_params,
                             r_dstr=vol_r_dstr, r_params=vol_r_params,
                             n_dstr=vol_nv_dstr, n_params=vol_nv_params,
                             batch_rng=rng)

    srfsampler = SphereSampler(batch_rng=rng)

    # %%
    #########################################################
    #### Evaluation Param Tensorization and Sanitization ####
    #########################################################

    # The following evaluates the 'eval/*' options and creates 
    # array parameters for evaluation.
    # The input is mainly the `evalcfgs` dictionary, which holds 
    # some keys and list/string values. The output will be the 
    # `evalprms` dictionary which has the same keys but with 
    # np.array values.
    # --------------
    # Example input: 
    #   evalcfgs = {'ur': {'dstr': 'uniform', 
    #       'low': [0], 
    #       'high': '[sqrt(dim)]'}}
    # --------------
    # Example output
    #   evalprms = {'ur': {'dstr': 'uniform',
    #       'low' : torch.tensor([0]).expand(n_seeds, dim)
    #       'high': torch.tensor(np.sqrt(dim)).expand(n_seeds, dim))}}

    dstr2shapes = {'uniform': {'low':  '(dim,)', 
                               'high': '(dim,)'},
                   'grid':    {'low':  '(dim,)', 
                               'high': '(dim,)'},
                   'ball':    {'c':    '(dim,)', 
                               'r':    '()'    },
                   'trnvol': {}}

    assert all('dstr' in eopts for eopts in evalcfgs.values())
    assert all('frq'  in eopts for eopts in evalcfgs.values())
    assert all('n'    in eopts for eopts in evalcfgs.values())
    evalprms = odict()
    for eid, eopts_ in evalcfgs.items():
        eopts = eopts_.copy()
        eparam = odict()
        for eopt in ('dstr', 'n', 'frq'):
            eparam[eopt] = eopts.pop(eopt)

        edstr = eparam['dstr']
        msg_  = f'Unknown eval "{eid}" dstr -> "{edstr}". '
        msg_ += f'dstr should be one of {dstr2shapes.keys()}.'
        assert edstr in dstr2shapes, msg_

        estore_dflt = (edstr == 'grid')
        eparam['store'] = eopts.pop('store', estore_dflt)

        opts2shape = dstr2shapes[edstr]
        for eopt, eoptshpstr in opts2shape.items():
            # Example: edstr = 'uniform'
            #          eopt = 'low'
            #          eoptshpstr = '(dim,)'
            #          eoptshp = (dim,)
            #          eoptval = "[sqrt(dim)]"
            #          eopt_pnp0 = np.array([sqrt(dim)]*dim)
            #          eopt_pnp0.shape = (dim,)
            #          eopt_pnp = eopt_pnp0.expand(n_seeds, dim)
            #          eopt_pnp.shape = (n_seeds, dim)
            #          eopt_p = torch.from_numpy(eopt_pa0)
            #          eopt_p.shape = (n_seeds, dim)
            msg_  = f'"eval/{eid}/{eopt}" must be fixed and determined.'
            msg_ += f' Hierarchical support not available yet.'
            assert eopt in eopts, msg_

            eoptval = eopts.pop(eopt)
            etrns = {eopt: eoptval, 'dim': dim, 
                     'sqrt': np.sqrt}

            eopt_pnp0 = get_arr(eopt, eoptshpstr, etrns)        
            eoptshp = eval_formula(eoptshpstr, {'dim': dim})
            eopt_pnp = np.broadcast_to(eopt_pnp0[None, ...],
                                       (n_seeds, *eoptshp)).copy()
            assert eopt_pnp.shape == (n_seeds, *eoptshp)
            eopt_pc = torch.from_numpy(eopt_pnp)
            eopt_p = eopt_pc.to(device=tch_device, dtype=tch_dtype)
            assert eopt_p.shape == (n_seeds, *eoptshp)
            eparam[eopt] = eopt_p

        assert len(eopts) == 0, f'unused eval items left: {eopts}'
        evalprms[eid] = eparam

    #########################################################
    ############# Evaluation Parameter Creation #############
    #########################################################
    for eid, eopts in evalprms.items():
        edstr = eopts['dstr']
        n_evlpnts = eopts['n']
        if edstr == 'uniform':
            e_low_ = eopts['low']
            assert e_low_.shape == (n_seeds, dim)
            e_low = e_low_.unsqueeze(dim=-2)
            assert e_low.shape == (n_seeds, 1, dim)

            e_high_ = eopts['high']
            assert e_high_.shape == (n_seeds, dim)
            e_high = e_high_.unsqueeze(dim=-2)
            assert e_high.shape == (n_seeds, 1, dim)

            e_slope = e_high - e_low
            assert e_slope.shape == (n_seeds, 1, dim)

            eopts['bias'] = e_low
            eopts['slope'] = e_slope
        elif edstr == 'ball':
            e_c_ = eopts['c']
            assert e_c_.shape == (n_seeds, dim)
            e_c = e_c_.unsqueeze(dim=-2).expand(n_seeds, n_evlpnts, dim)
            assert e_c.shape == (n_seeds, n_evlpnts, dim)

            e_r_ = eopts['r']
            assert e_r_.shape == (n_seeds,)

            e_r = e_r_.reshape(n_seeds, 1, 1).expand(n_seeds, n_evlpnts, 1)
            assert e_r.shape == (n_seeds, n_evlpnts, 1)

            eopts['c_xpnd'] = e_c
            eopts['r_xpnd'] = e_r
        elif edstr == 'trnvol':
            pass
        elif edstr == 'grid':
            n_g_ = eopts['n']
            if isinstance(n_g_, int):
                n_gpd, n_g = [int(np.ceil(n_g_**(1./dim)))] * dim, n_g_
                assert n_g == int(np.prod(n_gpd))
            elif isinstance(n_g_, (list, tuple)):
                n_gpd = n_g_
                n_g = int(np.prod(n_gpd))
            else:
                raise ValueError(f'eopts[\'n\']={n_g} is not defined')

            elowt = eopts['low']
            assert elowt.shape == (n_seeds, dim)

            ehight = eopts['high']
            assert ehight.shape == (n_seeds, dim)

            assert (elowt[:1] == elowt).all()
            assert (ehight[:1] == ehight).all()

            elow = elowt.cpu().detach().numpy()[0].tolist()
            ehigh = ehight.cpu().detach().numpy()[0].tolist()

            e_pntsplt = create_xg(elow, ehigh, n_gpd, tch_device, tch_dtype)
            assert e_pntsplt.shape == (*n_gpd, dim)

            e_pnts = e_pntsplt.reshape(1, n_g, dim).expand(n_seeds, n_g, dim)
            assert e_pnts.shape == (n_seeds, n_g, dim)

            eopts['pnts'] = e_pnts.to(tch_device, tch_dtype)
            eopts['pnts_plt'] = e_pntsplt.to(tch_device, tch_dtype)
            eopts['n_gpd'] = n_gpd
            eopts['n'] = n_g
        else:
            raise ValueError(f'"{edstr}" not defined')

    # %%
    #########################################################
    #### Collecting the Config Columns in the Dataframe #####
    #########################################################
    # Identifying the hyper-parameter from etc config columns
    hppats = ['problem', 'dim', 'vol/n', 'srfpts/n/mdl', 
        'srfpts/n/trg',  'srfpts/detspc', 'srfpts/dblsmpl',
        'trg/w', 'trg/btstrp', 'trg/tau', 'trg/reg/w', 'opt/lr', 
        'opt/dstr',  'nn/*', 'wire/*', 'ic/*', 'vol/*', 'eval/*', 'srch/*']
    etcpats = ['desc', 'date', 'opt/epoch', 'rng_seed/list', 'io/*']

    hpopts = [x for pat in hppats for x in 
               fnmatch.filter(cfg_dict_input.keys(), pat)]
    etcopts = [x for pat in etcpats for x in 
               fnmatch.filter(cfg_dict_input.keys(), pat)]

    err_list = []
    for opt in cfg_dict_input:
        if (opt in hpopts) and (opt in etcopts):
            msg_ = f'"{opt}" should both be treated as hp and etc!'
            err_list.append(msg_)
        if (opt not in hpopts) and (opt not in etcopts):
            msg_ = f'"{opt}" is neither hp nor etc!'
            err_list.append(msg_)
    if len(err_list) > 0:
        raise RuntimeError(('\n'+80*'*'+'\n').join(err_list))

    # Converting the list and tuples to strings
    hp_dict_ = odict()
    etc_dict_ = odict()
    for opt, val in cfg_dict_input.items():
        val = cfg_dict_input[opt]
        if isinstance(val, (int, float, str, bool, type(None))):
            srlval = val
        elif isinstance(val, (list, tuple)):
            srlval = repr(val)
        else: 
            msg_  = f'Not sure how to log "{opt}" with '
            msg_ += f'a value type of "{type(val)}"'
            raise RuntimeError(msg_)

        if opt in hpopts:
            hp_dict_[opt] = srlval
        elif opt in etcopts:
            etc_dict_[opt] = srlval
        else:
            raise RuntimeError(f'Not sure how to log "{opt}"')

    # Few exceptions for the etc directory
    etc_dict_['hostname'] = hostname
    etc_dict_['commit'] = commit_hash
    etc_dict_['date/cfg'] = etc_dict_.pop('date')
    etc_dict_['date/run'] = dtnow
    etc_dict_['io/dvc_mdl'] = tch_dvcmdl
    etc_dict_.pop('io/results_dir')
    etc_dict_.pop('io/storage_dir')

    # Repeating the values by n_seeds
    hp_dict = odict()
    for opt, val in hp_dict_.items():
        hp_dict[opt] = [val] * n_seeds
    etc_dict = odict()
    for opt, val in etc_dict_.items():
        etc_dict[opt] = [val] * n_seeds

# %% [markdown]
# # Training Loop

    # %%
    if results_dir is not None:
        pathlib.Path(os.sep.join([results_dir, cfg_tree])
                     ).mkdir(parents=True, exist_ok=True)
    if storage_dir is not None:
        cfgstrgpnt_dir = os.sep.join([storage_dir, cfg_tree, cfg_name])
        pathlib.Path(cfgstrgpnt_dir).mkdir(parents=True, exist_ok=True)
        strgidx = sum(isdir(f'{cfgstrgpnt_dir}/{x}') for x in os.listdir(cfgstrgpnt_dir))
        dtnow_ = dtnow[2:].replace('-', '').replace(':', '').replace('.', '')
        cfgstrg_dir = f'{cfgstrgpnt_dir}/{strgidx:02d}_{dtnow_}'
        pathlib.Path(cfgstrg_dir).mkdir(parents=True, exist_ok=True)
    if do_logtb:
        if 'tbwriter' in locals():
            tbwriter.close()
        tbwriter = SummaryWriter(cfgstrg_dir)
    if do_profile:
        profiler = Profiler()
        profiler.start()

    # %%
    # Initializing the model
    model = bffnn(dim, nn_width, nn_hidden, nn_act, (n_seeds,), rng, out_width=dim)
    if do_bootstrap:
        target = bffnn(dim, nn_width, nn_hidden, nn_act, (n_seeds,), rng, out_width=dim)
        target.load_state_dict(model.state_dict())
    else:
        target = model

    # Set the optimizer
    if opt_type == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr)
    elif opt_type == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr)
    else:
        raise NotImplementedError(f'opt/dstr="{opt_type}" not implmntd')

    # Evaluation tools
    erng = rng
    last_perfdict = dict()
    ema = EMA(gamma=0.999, gamma_sq=0.998)
    trn_sttime = time.time()

    # Data writer construction
    hdfpth = None
    if results_dir is not None:
        hdfpth = f'{results_dir}/{cfg_tree}/{cfg_name}.h5'
    avg_history = odict()
    dwriter = DataWriter(flush_period=ioflsh_period*n_seeds, 
                         compression_level=io_cmprssnlvl)

    if storage_dir is not None:
        with plt.style.context('default') as aa, plt.ioff() as bb:
            ax_rows, ax_cols = 7, 3
            figax_list = [plt.subplots(ax_rows, ax_cols, 
                figsize=(2.5*ax_cols, 2.5*ax_rows), dpi=100) for _ in range(3)]
            (fig_mdl, ax_mdl), (fig_trg, ax_trg), (fig_gt, ax_gt) = figax_list
            cax_mdl, cax_trg, cax_gt = None, None, None
            # cax_list = [[make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05) 
            #              for ax in np.array(axes).ravel().tolist()]
            #             for axes in (ax_mdl, ax_trg, ax_gt)]
            # cax_mdl, cax_trg, cax_gt = cax_list
        stat_history = defaultdict(list)
        model_history = odict()
        target_history = odict()
        
        if (srch_dstr is not None) and (dim == 2):
            with plt.style.context('default'):
                fig_srch, ax_srch = plt.subplots(1, 1, figsize=(3.2, 2.5), dpi=100)

    # %%
    for epoch in range(n_epochs+1):
        if srch_dstr is not None:
            update_problem = (epoch >= srch_inittrn) and ((epoch - srch_inittrn) % srch_frq == 0)
            if update_problem:
                if srch_dstr in ('mcmc',):
                    prpsd_srch_mu = old_srch_mu + srch_sigma * rng.normal((n_seeds, sdim))
                    assert prpsd_srch_mu.shape == (n_seeds, sdim)
                    
                    srch_mu_tch = prpsd_srch_mu
                    srch_mu = srch_mu_tch.detach().cpu().numpy()
                    assert srch_mu.shape == (n_seeds, sdim)
                    
                    prblmparams = mcmc_enc(srch_mu_tch)
                    problem = DeltaLineProblem(source=prblmparams['source'], 
                        sink=prblmparams['sink'], current=prblmparams['current'],
                        tch_device=tch_device, tch_dtype=tch_dtype)
                else:
                    raise ValueError(f'srch_dstr={srch_dstr} not implemented')
                
        opt.zero_grad()

        # Sampling the volumes
        volsamps = volsampler(n=n_srf)

        # Sampling the points from the srferes
        srfsamps = srfsampler(volsamps, n_points, do_detspacing=do_detspacing)
        points = nn.Parameter(srfsamps['points'])
        surfacenorms = srfsamps['tangents']
        areas = srfsamps['areas']
        assert points.shape == (n_seeds, n_srf, n_points, dim)
        assert surfacenorms.shape == (n_seeds, n_srf, n_points, dim)
        assert areas.shape == (n_seeds, n_srf,)

        points_mdl = points[:, :, :n_srfpts_mdl, :]
        assert points_mdl.shape == (n_seeds, n_srf, n_srfpts_mdl, dim)
        points_trg = points[:, :, n_srfpts_mdl:, :]
        assert points_trg.shape == (n_seeds, n_srf, n_srfpts_trg, dim)

        surfacenorms_mdl = surfacenorms[:, :, :n_srfpts_mdl, :]
        assert surfacenorms_mdl.shape == (n_seeds, n_srf, n_srfpts_mdl, dim)
        surfacenorms_trg = surfacenorms[:, :, n_srfpts_mdl:, :]
        assert surfacenorms_trg.shape == (n_seeds, n_srf, n_srfpts_trg, dim)

        # Making surface integral predictions using the reference model
        u_mdl = model(points_mdl)
        assert u_mdl.shape == (n_seeds, n_srf, n_srfpts_mdl, dim)
        nabla_x_u_mdl = curl(u_mdl, points_mdl, create_graph=True)
        assert nabla_x_u_mdl.shape == (n_seeds, n_srf, n_srfpts_mdl, dim)
        
        normprods_mdl = (nabla_x_u_mdl * surfacenorms_mdl).sum(dim=-1)
        assert normprods_mdl.shape == (n_seeds, n_srf, n_srfpts_mdl)
        if n_srfpts_mdl > 0:
            mean_normprods_mdl = normprods_mdl.mean(dim=-1, keepdim=True)
            assert mean_normprods_mdl.shape == (n_seeds, n_srf, 1)
        else:
            mean_normprods_mdl = 0.0

        # Making surface integral predictions using the target model
        u_trg = target(points_trg)
        assert u_trg.shape == (n_seeds, n_srf, n_srfpts_trg, dim)
        nabla_x_u_trg = curl(u_trg, points_trg, create_graph=not(do_bootstrap))
        assert nabla_x_u_trg.shape == (n_seeds, n_srf, n_srfpts_trg, dim)

        normprods_trg = (nabla_x_u_trg * surfacenorms_trg).sum(dim=-1)
        assert normprods_trg.shape == (n_seeds, n_srf, n_srfpts_trg)
        if do_dblsampling:
            assert n_rsdls == 2

            mean_normprods_trg1 = normprods_trg[..., 0::2].mean(
                dim=-1, keepdim=True)
            assert mean_normprods_trg1.shape == (n_seeds, n_srf, 1)

            mean_normprods_trg2 = normprods_trg[..., 1::2].mean(
                dim=-1, keepdim=True)
            assert mean_normprods_trg2.shape == (n_seeds, n_srf, 1)

            mean_normprods_trg = torch.cat(
                [mean_normprods_trg1, mean_normprods_trg2], dim=-1)
            assert mean_normprods_trg.shape == (n_seeds, n_srf, n_rsdls)
        else:
            assert n_rsdls == 1

            mean_normprods_trg = normprods_trg.mean(dim=-1, keepdim=True)
            assert mean_normprods_trg.shape == (n_seeds, n_srf, n_rsdls)

        # Linearly combining the reference and target predictions
        mean_normprods = (       w_trg  * mean_normprods_trg +
                          (1.0 - w_trg) * mean_normprods_mdl)
        assert mean_normprods.shape == (n_seeds, n_srf, n_rsdls)

        # Considering the surface areas
        pred_surfintegs = mean_normprods * areas.reshape(n_seeds, n_srf, 1)
        assert pred_surfintegs.shape == (n_seeds, n_srf, n_rsdls)

        # Getting the reference volume integrals
        ref_volintegs = problem.integrate_volumes(volsamps)
        assert ref_volintegs.shape == (n_seeds, n_srf)

        # Getting the residual terms
        resterms = pred_surfintegs - ref_volintegs.reshape(n_seeds, n_srf, 1)
        assert resterms.shape == (n_seeds, n_srf, n_rsdls)

        # Multiplying the residual terms
        if do_dblsampling:
            resterms_prod = resterms.prod(dim=-1)
            assert resterms_prod.shape == (n_seeds, n_srf)
        else:
            resterms_prod = torch.square(resterms).squeeze(-1)
            assert resterms_prod.shape == (n_seeds, n_srf)

        # Computing the main loss
        loss_main = resterms_prod.mean(-1)
        assert loss_main.shape == (n_seeds,)

        if do_bootstrap:
            with torch.no_grad():
                u_mdl_prime = target(points_mdl)
            loss_trgreg = torch.square(u_mdl - u_mdl_prime).mean([-3, -2, -1])
            assert loss_trgreg.shape == (n_seeds,)
        else:
            loss_trgreg = torch.zeros(n_seeds, device=tch_device, dtype=tch_dtype)
            assert loss_trgreg.shape == (n_seeds,)

        # The initial condition loss
        renew_ic = (epoch == 0) if (ic_frq == 0) else (epoch % ic_frq == 0)
        if ic_needsampling and renew_ic and (ic_dstr is not None):
            with torch.no_grad():
                if ic_dstr == 'sphere':
                    ic_normsamps =rng.normal((n_seeds, ic_n, dim))
                    assert ic_normsamps.shape == (n_seeds, ic_n, dim)

                    ic_ptstilde = ic_normsamps / ic_normsamps.norm()
                    assert ic_ptstilde.shape == (n_seeds, ic_n, dim)

                    assert ic_c.shape == (n_seeds, ic_n, dim)
                    assert ic_r.shape == (n_seeds, ic_n, 1)

                    ic_allpnts = ic_c + ic_ptstilde * ic_r
                    assert ic_allpnts.shape == (n_seeds, ic_n, dim)
                elif ic_dstr == 'trnvol':
                    icvols = volsampler(n=ic_n)
                    assert icvols['type'] == 'ball'

                    ic_c = icvols['centers']
                    assert ic_c.shape == (n_seeds, ic_n, dim)

                    ic_r_ = icvols['radii']
                    assert ic_r_.shape == (n_seeds, ic_n)

                    ic_r = ic_r_.unsqueeze(dim=-1)
                    assert ic_r.shape == (n_seeds, ic_n, 1)

                    untrd = rng.uniform((n_seeds, ic_n, 1))
                    assert untrd.shape == (n_seeds, ic_n, 1)

                    untr = untrd.pow(1.0 / dim)
                    assert untr.shape == (n_seeds, ic_n, 1)

                    ic_pntrs = untr * ic_r
                    assert ic_pntrs.shape == (n_seeds, ic_n, 1)

                    ic_theta = rng.normal((n_seeds, ic_n, dim))
                    assert ic_theta.shape == (n_seeds, ic_n, dim)

                    ic_thtilde = ic_theta / ic_theta.norm(dim=-1, keepdim=True)
                    assert ic_thtilde.shape == (n_seeds, ic_n, dim)

                    ic_allpnts = ic_c + ic_thtilde * ic_pntrs
                    assert ic_allpnts.shape == (n_seeds, ic_n, dim)
                else:
                    raise ValueError(f'ic/dstr={ic_dstr} not defined')

                ic_allgtvs = get_prob_sol(problem, ic_allpnts, eval_bs, 
                    get_field=False, out_lib='torch')['v']
                assert ic_allgtvs.shape == (n_seeds, ic_n)

        if ic_needsampling:
            ic_idxs = ((np.arange(ic_bs) + epoch * ic_bs) % ic_n).tolist()

            ic_pnts = ic_allpnts[:, ic_idxs, :]
            assert ic_pnts.shape == (n_seeds, ic_bs, dim)

            ic_vpreds = model(ic_pnts).squeeze(dim=-1)
            assert ic_vpreds.shape == (n_seeds, ic_bs)

            ic_gtvs = ic_allgtvs[:, ic_idxs]
            assert ic_gtvs.shape == (n_seeds, ic_bs)
        elif (ic_dstr is not None):
            ic_pnts = points_mdl.reshape(n_seeds, n_srf * n_srfpts_mdl, dim)
            assert ic_pnts.shape == (n_seeds, ic_bs, dim)

            ic_vpreds = u_mdl.reshape(n_seeds, n_srf * n_srfpts_mdl)
            assert ic_vpreds.shape == (n_seeds, ic_bs)

            with torch.no_grad():
                ic_gtvs = get_prob_sol(problem, ic_pnts, eval_bs, 
                    get_field=False, out_lib='torch')['v']
            assert ic_gtvs.shape == (n_seeds, ic_bs)

        if ic_bpp == 'bias':
            mdl_bias = model.layer_last[1].squeeze(dim=-1)
            assert mdl_bias.shape == (n_seeds, 1)

            ic_vpreds = ic_vpreds.detach() - mdl_bias.detach() + mdl_bias
            assert ic_vpreds.shape == (n_seeds, ic_bs)
        elif ic_bpp == 'all':
            pass
        else:
            raise RuntimeError(f'ic/bpp={ic_bpp} not defined')

        if ic_dstr is not None:
            loss_ic = torch.square(ic_vpreds - ic_gtvs).mean(dim=-1)
            assert loss_ic.shape == (n_seeds,)
        else:
            loss_ic = torch.zeros(n_seeds, dtype=tch_dtype, device=tch_device)
            assert loss_ic.shape == (n_seeds,)

        # The total loss
        loss = loss_main + w_trgreg * loss_trgreg + w_ic * loss_ic
        assert loss.shape == (n_seeds,)

        loss_sum = loss.sum()
        loss_sum.backward()

        # We will not update in the first epoch so that we will 
        # record the initialization statistics as well. Instead, 
        # we will update an extra epoch at the end.
        if (epoch > 0):
            opt.step()

        # Updating the target network
        if do_bootstrap and (epoch > 0):
            model_sd = model.state_dict()
            target_sd = target.state_dict()
            newtrg_sd = dict()
            with torch.no_grad():
                for key, param in model_sd.items():
                    param_trg = target_sd[key]
                    newtrg_sd[key] = tau * param_trg + (1-tau) * param
            target.load_state_dict(newtrg_sd)

        # computing the normal product variances
        with torch.no_grad(): 
            normprods = torch.cat([normprods_mdl, normprods_trg], dim=-1)
            npvm = (normprods.var(dim=-1)*areas.square()).mean(-1)

        # evaluating the performance of the model and target    
        perf_dict = dict()
        eval_strg = dict()
        for eid, eopts in evalprms.items():
            edstr = eopts['dstr']
            n_evlpnts = eopts['n']
            e_frq = eopts['frq']
            e_store = eopts['store']

            if (epoch % e_frq) > 0:
                assert eid in last_perfdict
                perf_dict[eid] = last_perfdict[eid]
                continue

            # Sampling the evaluation points
            with torch.no_grad():
                if edstr == 'uniform':
                    e_bias = eopts['bias']
                    assert e_bias.shape == (n_seeds, 1, dim)

                    e_slope = eopts['slope']
                    assert e_slope.shape == (n_seeds, 1, dim)

                    e_unfpnts = erng.uniform((n_seeds, n_evlpnts, dim))
                    assert e_unfpnts.shape == (n_seeds, n_evlpnts, dim)

                    e_pnts = e_bias + e_unfpnts * e_slope
                    assert e_pnts.shape == (n_seeds, n_evlpnts, dim)
                elif edstr in ('ball', 'trnvol'):
                    if edstr == 'ball':
                        e_c = eopts['c_xpnd']
                        assert e_c.shape == (n_seeds, n_evlpnts, dim)

                        e_r = eopts['r_xpnd']
                        assert e_r.shape == (n_seeds, n_evlpnts, 1)
                    elif edstr == 'trnvol':
                        evols = volsampler(n=n_evlpnts)
                        assert evols['type'] == 'disk'
                        assert dim == 3

                        e_c = evols['centers']
                        assert e_c.shape == (n_seeds, n_evlpnts, dim)
                        
                        e_n_ = evols['normals'].squeeze(-1)
                        assert e_n_.shape == (n_seeds, n_evlpnts, dim)

                        e_r_ = evols['radii']
                        assert e_r_.shape == (n_seeds, n_evlpnts)

                        e_r = e_r_.unsqueeze(dim=-1)
                        assert e_r.shape == (n_seeds, n_evlpnts, 1)
                    else:
                        raise RuntimeError(f'case not defined')

                    untrd = erng.uniform((n_seeds, n_evlpnts, 1))
                    assert untrd.shape == (n_seeds, n_evlpnts, 1)

                    untr = untrd.pow(1.0 / dim)
                    assert untr.shape == (n_seeds, n_evlpnts, 1)

                    e_pntrs = untr * e_r
                    assert e_pntrs.shape == (n_seeds, n_evlpnts, 1)

                    etheta = erng.normal((n_seeds, n_evlpnts, dim))
                    assert etheta.shape == (n_seeds, n_evlpnts, dim)

                    if edstr == 'trnvol':
                        etheta = etheta -  (etheta * e_n_).sum(dim=-1, keepdim=True)
                        assert etheta.shape == (n_seeds, n_evlpnts, dim)
                    
                    ethtilde = etheta / etheta.norm(dim=-1, keepdim=True)
                    assert ethtilde.shape == (n_seeds, n_evlpnts, dim)

                    e_pnts = e_c + ethtilde * e_pntrs
                    assert e_pnts.shape == (n_seeds, n_evlpnts, dim)
                elif edstr in ('grid'):
                    e_pnts = eopts['pnts']
                    assert e_pnts.shape == (n_seeds, n_evlpnts, dim)
                else:
                    raise RuntimeError(f'eval dstr "{edstr}" not implmntd')

            # Computing the model, target and ground truth solutions
            prob_sol = get_prob_sol(problem, e_pnts, n_eval=eval_bs, 
                get_field=False, out_lib='torch')

            e_prbsol = prob_sol['v']
            assert e_prbsol.shape == (n_seeds, n_evlpnts, dim)

            # Computing the model solution
            with torch.no_grad():
                mdl_sol = get_nn_sol(model, e_pnts, n_eval=eval_bs,
                    get_field=False, out_lib='torch')

            e_mdlsol = mdl_sol['v']
            assert e_mdlsol.shape == (n_seeds, n_evlpnts, dim)
            
            # Computing the target solution
            if do_bootstrap:
                with torch.no_grad():
                    trg_sol = get_nn_sol(target, e_pnts, n_eval=eval_bs, 
                        get_field=False, out_lib='torch')

                e_trgsol = trg_sol['v']
                assert e_trgsol.shape == (n_seeds, n_evlpnts, dim)

            eperfs = dict()
            eperfs['mdl'] = get_perfdict(e_pnts, e_mdlsol, e_prbsol)
            if do_bootstrap:
                eperfs['trg'] = get_perfdict(e_pnts, e_trgsol, e_prbsol)
            eperfs = deep2hie(eperfs, dictcls=dict)
            # Example: eperfs = {'mdl/pln/mse': ...,
            #                    'mdl/pln/mae': ...,
            #                    'mdl/bc/mse': ...,
            #                    'mdl/bc/mae': ...,
            #                    'mdl/slc/mse': ...,
            #                    'mdl/slc/mae': ...,
            #                    'trg/pln/mse': ...,
            #                    'trg/pln/mae': ...,
            #                    'trg/bc/mse': ...,
            #                    'trg/bc/mae': ...,
            #                    'trg/slc/mse': ...,
            #                    'trg/slc/mae': ...,
            #                   }
            perf_dict[eid] = eperfs
            last_perfdict[eid] = eperfs
            
            if do_logtb:
                for kk, vv in eperfs.items():
                    tbwriter.add_scalar(f'perf/{eid}/{kk}', vv.mean(), epoch)
            
            # Storing the evaluation results
            if e_store:
                e_strg = dict()
                e_strg['sol/mdl'] = e_mdlsol
                if do_bootstrap:
                    e_strg['sol/trg'] = e_trgsol
                e_strg['sol/gt'] = e_prbsol
                if edstr != 'grid':
                    e_strg['pnts'] = e_pnts
                e_strg = {kk: vv.detach().cpu().numpy().astype(np.float16) 
                          for kk, vv in e_strg.items()}
                eval_strg[eid] = e_strg

                if do_logtb and (edstr == 'grid') and (dim == 3):
                    soltd_list = [('mdl', mdl_sol, fig_mdl, ax_mdl, cax_mdl)]
                    if do_bootstrap:
                        soltd_list += [('trg', trg_sol, fig_trg, ax_trg, cax_trg)]
                    soltd_list += [('gt', prob_sol, fig_gt, ax_gt, cax_gt)]
                    for sol_t, sol_dict, fig, ax, cax in soltd_list:
                        e_ngpd = eopts['n_gpd']
                        with torch.no_grad():
                            y_plt1 = sol_dict['v']
                            e_ng = int(np.prod(e_ngpd))
                            assert y_plt1.shape == (n_seeds, e_ng, dim)
                            y_plt2 = y_plt1 - y_plt1.mean(dim=-2, keepdim=True)
                            assert y_plt2.shape == (n_seeds, e_ng, dim)
                            y_plt3 = y_plt2.mean(dim=0)
                            assert y_plt3.shape == (e_ng, dim)
                        draw_heatmap(eopts['pnts_plt'], y_plt3, fig, ax,
                            field_abrv='A', components='xyz', zstr_loc='left', 
                            show_title='all')
                        fig.set_tight_layout(True)
                        tbwriter.add_figure(f'viz/{eid}/{sol_t}', fig, epoch)
                    tbwriter.flush()
        
        # Applying search on the latent problem parameters
        update_search = False
        if srch_dstr is not None:  
            update_search = ((epoch - srch_inittrn) % srch_frq == (srch_frq - 1))
            update_search = update_search and (epoch >= srch_inittrn)
        
        if update_search:
            assert srch_dstr == 'mcmc'
        
            need_field = (obs_y_dstr == 'field')
            sol_key = {'potential': 'v', 'field': 'e'}[obs_y_dstr]
            
            # Computing the problem solution
            obs_probsol = get_prob_sol(problem, obs_x, n_eval=eval_bs, 
                get_field=need_field, out_lib='torch')[sol_key]
            if (obs_y_dstr == 'potential'):
                obs_probsol = obs_probsol.unsqueeze(-1)
                obs_probsol = obs_probsol - obs_probsol.mean(dim=-2, keepdim=True)
            assert obs_probsol.shape == (n_seeds, obs_n, obs_ydim)

            # Computing the model solution
            obs_mdlsol = get_nn_sol(model, obs_x, n_eval=eval_bs,
                get_field=need_field, out_lib='torch')[sol_key]
            if (obs_y_dstr == 'potential'):
                obs_mdlsol = obs_mdlsol.unsqueeze(-1)
                obs_mdlsol = obs_mdlsol - obs_mdlsol.mean(dim=-2, keepdim=True)
            assert obs_mdlsol.shape == (n_seeds, obs_n, obs_ydim)
            
            with torch.no_grad():
                obs_mdlerr = obs_mdlsol - obs_y
                assert obs_mdlerr.shape == (n_seeds, obs_n, obs_ydim)
                
                assert srch_metric_dstr == 'mse'
                
                obs_loss = obs_mdlerr.square().sum(dim=-1)
                assert obs_loss.shape == (n_seeds, obs_n)
                    
                if (srch_metric_min is not None) or (srch_metric_max is not None):
                    obs_lossclp = torch.clip(obs_loss, srch_metric_min, srch_metric_max)
                    assert obs_lossclp.shape == (n_seeds, obs_n)
                else:
                    obs_lossclp = obs_loss
                    assert obs_lossclp.shape == (n_seeds, obs_n)
                
                prpsd_loglike = obs_lossclp.mean(dim=-1) * (-srch_metric_coeff)
                assert prpsd_loglike.shape == (n_seeds,)
                
                prpsd_logpri_ = normal_logprob(prpsd_srch_mu, srch_prior_loc_tch, srch_prior_scale_tch)
                assert prpsd_logpri_.shape == (n_seeds, srch_n)
                
                prpsd_logpri = prpsd_logpri_.sum(dim=-1)
                assert prpsd_logpri.shape == (n_seeds,)
                
                prpsd_logpost = prpsd_logpri + prpsd_loglike
                assert prpsd_logpost.shape == (n_seeds,)
                
                if old_logpri is None:
                    old_logpri = prpsd_logpri
                    assert old_logpri.shape == (n_seeds,)
                
                    old_loglike = prpsd_loglike
                    assert old_loglike.shape == (n_seeds,)
                    
                    old_logpost = prpsd_logpost
                    assert old_logpost.shape == (n_seeds,)
                    
                    best_loglike = old_loglike
                    assert best_loglike.shape == (n_seeds,)

                delta_logpost = prpsd_logpost - old_logpost
                assert delta_logpost.shape == (n_seeds,)
                
                acceptance = (srch_tmprtr * delta_logpost) > rng.uniform((n_seeds, 1)).log().squeeze(-1)
                assert acceptance.shape == (n_seeds,)
                
                # Updating the monte carlo chains and log-probabilities
                old_srch_mu = torch.where(acceptance.reshape(n_seeds, 1, 1), prpsd_srch_mu, old_srch_mu)
                assert old_srch_mu.shape == (n_seeds, sdim)
                
                old_loglike = torch.where(acceptance, prpsd_loglike, old_loglike)
                assert old_loglike.shape == (n_seeds,)
                
                old_logpri = torch.where(acceptance, prpsd_logpri, old_logpri)
                assert old_logpri.shape == (n_seeds,)
                
                old_logpost =  torch.where(acceptance, prpsd_logpost, old_logpost)
                assert old_logpost.shape == (n_seeds,)
                
                best_loglike = torch.where(old_loglike > best_loglike, old_loglike, best_loglike)
                assert best_loglike.shape == (n_seeds,)
                
                # Applying the reset curriculum
                i_srchdrwas = (epoch - srch_inittrn) // srch_frq
                
                if (i_srchdrwas % (n_srchdraws // srch_reset_n) == 0):
                    old_srch_mu_ = old_srch_mu.reshape(n_grps, n_cpg, sdim)
                    assert old_srch_mu_.shape == (n_grps, n_cpg, sdim)
                    
                    old_logpost_ = old_logpost.reshape(n_grps, n_cpg)
                    assert old_logpost_.shape == (n_grps, n_cpg)
                    
                    top_idx = torch.topk(old_logpost_, srch_reset_k, dim=1, largest=True, sorted=True).indices
                    assert top_idx.shape == (n_grps, srch_reset_k)
                    
                    top_chains = torch.take_along_dim(old_srch_mu_, top_idx.unsqueeze(-1).unsqueeze(-1), dim=-3)
                    assert top_chains.shape == (n_grps, srch_reset_k, sdim)
                    
                    # resetting the mu
                    srch_reset_reps = n_cpg // srch_reset_k
                    old_srch_mu = top_chains.reshape(n_grps, 1, srch_reset_k, sdim)
                    old_srch_mu = old_srch_mu.expand(n_grps, srch_reset_reps, srch_reset_k, sdim)
                    old_srch_mu = old_srch_mu.reshape(n_seeds, sdim)
                    assert old_srch_mu.shape == (n_seeds, sdim)
                    
                    # resetting the models
                    mdlrepsd = {name: replicate_top(param, top_idx) 
                                for name, param in model.state_dict().items()}
                    model.load_state_dict(mdlrepsd)

                    if do_bootstrap:
                        trgrepsd = {name: replicate_top(param, top_idx) 
                            for name, param in target.state_dict().items()}
                        target.load_state_dict(trgrepsd)
                    
                    # resetting the optimizer
                    if opt_type == 'sgd':
                        pass
                    elif opt_type == 'adam':
                        optrepsd = deepcopy(opt.state_dict())
                        for kk, vv in optrepsd['state'].items():
                            vv['exp_avg'] = replicate_top(vv['exp_avg'], top_idx)
                            vv['exp_avg_sq'] = replicate_top(vv['exp_avg_sq'], top_idx)
                        opt.load_state_dict(optrepsd)
                    else:
                        raise ValueError(f'opt_type={opt_type} not defined')
                
                if do_logtb:
                    blm = best_loglike.reshape(n_grps, n_cpg)
                    tbwriter.add_scalar('search/acceptance', acceptance.float().mean(), epoch)
                    for q in np.linspace(0, 1, 5):
                        tbwriter.add_scalar(f'search/best_loglike/q{int(100*q):02d}', 
                                            blm.quantile(q, dim=-1).median(), epoch)
                    tbwriter.add_scalar('search/loglike', old_loglike.mean(), epoch)
                    tbwriter.add_scalar('search/logpri', old_logpri.mean(), epoch)
                    tbwriter.add_scalar('search/logpost', old_logpost.mean(), epoch)

                    if (srch_enc_dstr == 'srcsnkxy'):
                        # Finding the best group for scatter plotting
                        i_grp = old_loglike.reshape(n_grps, n_cpg).max(dim=-1).values.argmax()
                        x_scatt = old_srch_mu.reshape(n_grps, n_cpg, wire_n, dim-1)[i_grp]
                        assert x_scatt.shape == (n_cpg, wire_n, dim-1)
                        x_scatt = x_scatt.reshape(n_cpg * wire_n, dim-1)
                        assert x_scatt.shape == (n_cpg * wire_n, dim-1)
                        x_scatt = x_scatt.detach().cpu().numpy()
                        assert x_scatt.shape == (n_cpg * wire_n, dim-1)

                        ax_srch.clear()
                        ax_srch.scatter(x_scatt[:, 0], x_scatt[:, 1], s=1)
                        
                        fig_srch.set_tight_layout(True)
                        tbwriter.add_figure(f'viz/srch/mu', fig_srch, epoch)
                    
                    tbwriter.flush()
  
        # monitoring the resource utilization 
        if epoch % iomon_period == 0:
            s_rsrc = resource.getrusage(resource.RUSAGE_SELF)
            c_rsrc = resource.getrusage(resource.RUSAGE_CHILDREN)
            
            psmem = psutil.virtual_memory()
            pscpu = psutil.cpu_times()
            pscpuload = psutil.getloadavg()
            mon_dict = {'cpu/mem/tot': [psmem.total] * n_seeds, 
                'cpu/mem/avail': [psmem.available] * n_seeds, 
                'cpu/mem/used': [psmem.used] * n_seeds,
                'cpu/mem/free': [psmem.free] * n_seeds,
                'cpu/time/user/ps': [pscpu.user] * n_seeds,
                'cpu/time/sys/ps': [pscpu.system] * n_seeds,
                'cpu/time/idle/ps': [pscpu.idle] * n_seeds,
                'cpu/load/1m': [pscpuload[0]] * n_seeds,
                'cpu/load/5m': [pscpuload[1]] * n_seeds,
                'cpu/load/15m': [pscpuload[2]] * n_seeds,
                'cpu/time/train': [time.time()   - trn_sttime] * n_seeds,
                'cpu/time/sys/py':   [s_rsrc.ru_stime  + c_rsrc.ru_stime] * n_seeds,
                'cpu/time/user/py':  [s_rsrc.ru_utime  + c_rsrc.ru_utime] * n_seeds,
                'n_seeds': [n_seeds] * n_seeds}
            if 'cuda' in device_name:
                t_gpumem = torch.cuda.get_device_properties(tch_device).total_memory
                r_gpumem = torch.cuda.memory_reserved(tch_device)
                a_gpumem = torch.cuda.memory_allocated(tch_device)
                f_gpumem = r_gpumem - a_gpumem
                mon_dict.update({'gpu/mem/tot':   [t_gpumem] * n_seeds,
                                 'gpu/mem/res':   [r_gpumem] * n_seeds,
                                 'gpu/mem/alloc': [a_gpumem] * n_seeds,
                                 'gpu/mem/free':  [f_gpumem] * n_seeds})

        # pushing the results to the data writer
        psld = deep2hie({'perf': perf_dict}, odict)
        slst = [('loss/total',  loss.tolist()),
                ('loss/main',   loss_main.tolist()),
                ('loss/trgreg', loss_trgreg.tolist()),
                ('loss/ic',     loss_ic.tolist()),
                ('npvm',        npvm.tolist()),
                *list(psld.items())]
        stat_dict = odict(slst)
        for stat_name, stat_vals in stat_dict.items():
            avg_history.setdefault(stat_name, [])
            avg_history[stat_name].append(stat_vals)

        dtups = []
        if epoch % io_avgfrq == 0:
            avg_statlst  = [('epoch',       [epoch] * n_seeds),
                            ('rng_seed',    rng_seeds.tolist())]
            avg_statlst += [(name, np.stack(svl, axis=0).mean(axis=0).tolist())
                             for name, svl in avg_history.items()]
            avg_statdict = odict(avg_statlst)

            dtups += [('hp',    hp_dict,      'pd.cat'),
                      ('stat',  avg_statdict, 'pd.qnt'),
                      ('etc',   etc_dict,     'pd.cat')]
            avg_history = odict()
            
        for eid, e_strg in eval_strg.items():
            msg_ =  f'eval/{eid} requires storage, thus "eval/{eid}/frq" '
            msg_ += f'% "io/avg/frq" == 0 should hold.'
            assert epoch % io_avgfrq == 0, msg_
            dtups += [(f'var/eval/{eid}', e_strg, 'np.arr')]
        
        if epoch % iomon_period == 0:
            assert epoch % io_avgfrq == 0
            dtups += [('mon', mon_dict, 'pd.qnt')]

        if epoch % chkpnt_period == 0:
            assert epoch % io_avgfrq == 0
            mdl_sdnp = {k: v.detach().cpu().numpy() 
                for k, v in model.state_dict().items()}
            dtups += [('mdl',   mdl_sdnp, 'np.arr')]
            if do_bootstrap:
                trg_sdnp = {k: v.detach().cpu().numpy() 
                    for k, v in target.state_dict().items()}
                dtups += [('trg',   trg_sdnp, 'np.arr')]
                
        if update_search:
            srch_strg = {'prpsd/mu': srch_mu_tch, 'prpsd/loglike': prpsd_loglike, 
                         'prpsd/logpri': prpsd_logpri, 'prpsd/logpost': prpsd_logpost,
                         'old/mu': old_srch_mu, 'old/loglike': old_loglike,
                         'old/logpri': old_logpri, 'old/logpost': old_logpost,
                         'acceptance': acceptance}
            srch_strg = {k: v.detach().cpu().numpy() for k,v in srch_strg.items()}
            dtups += [(f'var/srch', srch_strg, 'np.arr')]

        dwriter.add(data_tups=dtups, file_path=hdfpth)

        # Computing the loss moving averages
        loss_ema_mean, loss_ema_std_mean = ema('loss', loss)
        npvm_ema_mean, npvm_ema_std_mean = ema('npvm', npvm)
        if (epoch % 1000 == 0) and (results_dir is not None):
            print_str = f'Epoch {epoch}, EMA loss = {loss_ema_mean:.4f}'
            print_str += f' +/- {2*loss_ema_std_mean:.4f}'
            print_str += f', EMA Field-Norm Product Variance = {npvm_ema_mean:.4f}'
            print_str += f' +/- {2*npvm_ema_std_mean:.4f} ({time.time()-trn_sttime:0.1f} s)'
            print(print_str, flush=True)

        if do_logtb:
            import logging
            logging.getLogger("tensorboardX.x2num").setLevel(logging.CRITICAL)  
            tbwriter.add_scalar('loss/total', loss.mean(), epoch)
            tbwriter.add_scalar('loss/main', loss_main.mean(), epoch)
            tbwriter.add_scalar('loss/trgreg', loss_trgreg.mean(), epoch)
            tbwriter.add_scalar('loss/ic', loss_ic.mean(), epoch)
            tbwriter.add_scalar('loss/npvm', npvm.mean(), epoch)

        if do_tchsave and (epoch % chkpnt_period == 0):
            model_history[epoch] = deepcopy({k: v.cpu() for k, v
                in model.state_dict().items()})
            target_history[epoch] = deepcopy({k: v.cpu() for k, v
                in target.state_dict().items()})
            

    if results_dir is not None:
        print(f'Training finished in {time.time() - trn_sttime:.1f} seconds.')
    dwriter.close()
    if do_logtb:
        tbwriter.flush()
    
    outdict = dict()
    tchmemusage = profmem()
    assert str(tch_device) in tchmemusage
    if 'cuda' in device_name:
        tch_dvcmem = torch.cuda.get_device_properties(tch_device).total_memory
    else:
        tch_dvcmem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    outdict['dvc/mem/alloc'] = tchmemusage[str(tch_device)]
    outdict['dvc/mem/total'] = tch_dvcmem


    # %%
    fig = None
    has_grid = any(eopts.get('dstr', '') == 'grid' 
                    for eid, eopts in evalprms.items())
    if (storage_dir is not None) and has_grid:
        eopts = list(eopts for eid, eopts in evalprms.items()
                        if eopts.get('dstr', '') == 'grid')[0]
        e_pnts = eopts['pnts']
        
        n_sols = 2 + do_bootstrap
        n_rows, n_cols = 7, 3
        fig, axes_all = plt.subplots(n_rows, n_cols*n_sols, 
            figsize=(n_cols*n_sols*2.5, n_rows*2.0), dpi=36)
        axes_all = np.array(axes_all).reshape(n_rows, n_cols*n_sols)
        figaxes_list = [(fig, axes_all[:, i*3:i*3+3]) for i in range(n_sols)]
        
        # Computing the model, target and ground truth solutions
        prob_sol = get_prob_sol(problem, e_pnts, n_eval=eval_bs, 
            get_field=False, out_lib='torch')
        with torch.no_grad():
            mdl_sol = get_nn_sol(model, e_pnts, n_eval=eval_bs, 
                get_field=False, out_lib='torch') 
            if do_bootstrap:
                trg_sol = get_nn_sol(target, e_pnts, n_eval=eval_bs, 
                    get_field=False, out_lib='torch')
                
        soltd_list = [('gt', prob_sol, figaxes_list[0], 'Ground Truth'),
                    ('mdl', mdl_sol, figaxes_list[1], 'Prediction')]
        if do_bootstrap:
            soltd_list += [('trg', trg_sol, figaxes_list[2], 'Target')]
            
        for sol_t, sol_dict, (fig, ax), ax_ttl in soltd_list:
            draw_heatmap(eopts['pnts_plt'], sol_dict['v'].mean(dim=0), fig, ax, 
                field_abrv=f'{ax_ttl} A', components='xyz', show_title='top', zstr_loc='left')
            fig.set_tight_layout(True)
        
    fig

    # %%
    if do_tchsave:
        torch.save(model_history, f'{cfgstrg_dir}/ckpt_mdl.pt')
        if do_bootstrap:
            torch.save(target_history, f'{cfgstrg_dir}/ckpt_trg.pt')
    if storage_dir is not None:
        shutil.copy2(hdfpth, f'{cfgstrg_dir}/progress.h5')
        if fig is not None:
            fig.savefig(f'{cfgstrg_dir}/finalpred.pdf', dpi=144, bbox_inches="tight")   
    if do_profile:
        profiler.stop()
        html = profiler.output_html()
        htmlpath = f'{cfgstrg_dir}/profiler.html'
        with open(htmlpath, 'w') as fp:
            fp.write(html.encode('ascii', errors='ignore').decode('ascii'))


    # %% tags=["active-py"]
    return outdict

    
if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('-c', '--configid', action='store', type=str, required=True)
        my_parser.add_argument('-d', '--device',   action='store', type=str, required=True)
        my_parser.add_argument('-s', '--nodesize', action='store', type=int, default=1)
        my_parser.add_argument('-r', '--noderank', action='store', type=int, default=0)
        my_parser.add_argument('-i', '--rsmindex', action='store', type=str, default="0.0")
        my_parser.add_argument('--dry-run', action='store_true')
        args = my_parser.parse_args()
        args_configid = args.configid
        args_device_name = args.device
        args_nodesize = args.nodesize
        args_noderank = args.noderank
        arsg_rsmindex = args.rsmindex
        args_dryrun = args.dry_run
    else:
        args_configid = 'lvl1/lvl2/poiss2d'
        args_device_name = 'cuda:0'
        args_nodesize = 1
        args_noderank = 0
        arsg_rsmindex = 0
        args_dryrun = True

    assert args_noderank < args_nodesize
    cfgidsplit = args_configid.split('/')
    # Example: args_configid == 'lvl1/lvl2/poiss2d'
    config_id = cfgidsplit[-1]
    # Example: config_id == 'poiss2d'
    config_tree = '/'.join(cfgidsplit[:-1])
    # Example: config_tree == 'lvl1/lvl2'

    import os
    os.makedirs(configs_dir, exist_ok=True)
    # Example: configs_dir == '.../code_bspinn/config'
    os.makedirs(results_dir, exist_ok=True)
    # Example: results_dir == '.../code_bspinn/result'
    # os.makedirs(storage_dir, exist_ok=True)
    # Example: storage_dir == '.../code_bspinn/storage'
    
    if args_dryrun:
        print('>> Running in dry-run mode', flush=True)

    cfg_path_ = f'{configs_dir}/{config_tree}/{config_id}'
    cfg_exts = [cfg_ext for cfg_ext in ['json', 'yml', 'yaml'] if exists(f'{cfg_path_}.{cfg_ext}')]
    assert len(cfg_exts) < 2, f'found multiple {cfg_exts} extensions for {cfg_path_}'
    assert len(cfg_exts) > 0, f'found no json or yaml config at {cfg_path_}'
    cfg_ext = cfg_exts[0]
    cfg_path = f'{cfg_path_}.{cfg_ext}'
    print(f'>> Reading configuration from {cfg_path}', flush=True)
    
    if cfg_ext.lower() == 'json':
        with open(cfg_path, 'r') as fp:
            json_cfgdict = json.load(fp, object_pairs_hook=odict)
    elif cfg_ext.lower() in ('yml', 'yaml'):
        with open(cfg_path, 'r') as fp:
            json_cfgdict = odict(yaml.safe_load(fp))
    else:
        raise RuntimeError(f'unknown config extension: {cfg_ext}')
    
    if args_dryrun:
        import tempfile
        temp_resdir = tempfile.TemporaryDirectory()
        temp_strdir = tempfile.TemporaryDirectory()
        print(f'>> [dry-run] Temporary results dir placed at {temp_resdir.name}')
        print(f'>> [dry-run] Temporary storage dir placed at {temp_strdir.name}')
        results_dir = temp_resdir.name
        storage_dir = temp_strdir.name
        
        dr_maxfrq = 10
        dr_opts = {'opt/epoch': dr_maxfrq, 'io/cmprssn_lvl': 0}
        for opt in dr_opts:
            assert opt in json_cfgdict
            json_cfgdict[opt] = dr_opts[opt]
        for opt in fnmatch.filter(json_cfgdict.keys(), '*/frq'):
            if json_cfgdict[opt] > dr_maxfrq:
                json_cfgdict[opt] = dr_opts[opt] = dr_maxfrq
        print(f'>> [dry-run] The following options were made overriden:', flush=True)
        for opt, val in dr_opts.items():
            print(f'>>           {opt}: {val}', flush=True)
            
        
    nodepstfx = '' if args_nodesize == 1 else f'_{args_noderank:02d}'
    # Example: nodepstfx in ('', '_01')
    json_cfgdict['io/config_id'] = f'{config_id}{nodepstfx}'
    # Example: ans in ('poiss2d', 'poiss2d_01')
    json_cfgdict['io/results_dir'] = f'{results_dir}/{config_tree}'
    # Example: ans == '.../code_bspinn/result/lvl1/lv2/poiss2d'
    json_cfgdict['io/storage_dir'] = None # f'{storage_dir}/{config_tree}'
    # Example: ans == '.../code_bspinn/storage/lvl1/lv2/poiss2d'
    json_cfgdict['io/tch/device'] = args_device_name
    # Example: args_device_name == 'cuda:0'

    # Pre-processing and applying the looping processes
    all_cfgdicts = preproc_cfgdict(json_cfgdict)
    
    # Selecting this node's config dict subset
    node_cfgdicts = [cfg for i, cfg in enumerate(all_cfgdicts) 
                     if (i % args_nodesize == args_noderank)]
    n_nodecfgs = len(node_cfgdicts)
    
    # Going over the config dicts one-by-one
    rsmidx, rsmprt = tuple(int(x) for x in arsg_rsmindex.split('.'))
    for cfgidx, config_dict in enumerate(node_cfgdicts):
        if cfgidx < rsmidx:
            continue
        # Getting a single seed run to estimate the memory usage
        tempcfg = config_dict.copy()
        tempcfg['io/results_dir'] = None
        tempcfg['io/storage_dir'] = None
        tempcfg['rng_seed/list'] = [0]
        tempcfg['opt/epoch'] = 0
        tod = main(tempcfg)
        allocmem, totmem = tod['dvc/mem/alloc'], tod['dvc/mem/total']
        nsd_max = int(0.5 * totmem / allocmem)
        
        # Computing how many parts we must split the original config into
        cfg_seeds = config_dict['rng_seed/list']
        nprts = int(np.ceil(len(cfg_seeds) / nsd_max))
        print(f'>> Config index {cfgidx} takes {allocmem/1e6:.1f} ' + 
              f'MB/seed (out of {totmem/1e9:.1f} GB)', flush=True)
        print(f'>> Config index {cfgidx} must be ' + 
              f'devided into {nprts} parts.', flush=True)
        
        # Looping over each part of the config
        for iprt in range(nprts):
            if (cfgidx == rsmidx) and (iprt < rsmprt):
                continue
            print(f'>>> Started Working on config index {cfgidx}.{iprt}' + 
                  f' (out of {nprts} parts and {n_nodecfgs} configs).', flush=True)
            iprtcfgseeds = cfg_seeds[(iprt*nsd_max):((iprt+1)*nsd_max)]
            iprtcfgdict = config_dict.copy()
            iprtcfgdict['rng_seed/list'] = iprtcfgseeds
            main(iprtcfgdict)
            print('-' * 40, flush=True)
        print(f'>> Finished working on config index {cfgidx} ' + 
              f'(out of {n_nodecfgs} configs).', flush=True)
        print('='*80, flush=True)
        
    if args_dryrun:
        print(f'>> [dry-run] Cleaning up {temp_resdir.name}')
        temp_resdir.cleanup()
        print(f'>> [dry-run] Cleaning up {temp_strdir.name}')
        temp_strdir.cleanup()
