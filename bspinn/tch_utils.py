# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3.8
#     language: python
#     name: p38
# ---

# %% [markdown]
# ## PyTorch Utilities

# %% code_folding=[5, 12, 42, 373]
import gc
import math
import numpy as np
import torch
import warnings
from torch import nn
from collections import defaultdict
from torch.nn.init import calculate_gain
from torch.nn.init import _calculate_correct_fan
from torch.nn.init import _calculate_fan_in_and_fan_out


def isscalar(v):
    if torch.is_tensor(v):
        return v.numel() == 1
    else:
        return np.isscalar(v)

def torch_qr_eff(a, mode='complete', out=None, gram='classical'):
    """
    Due to a bug in MAGMA, qr on cuda is super slow for small matrices. 
    Therefore, this step must be performed on the cpu.
    
    This function aims to provide a temporary relief for using 
    `torch.linalg.qr` on GPU by implementing a Gram-Schmidt process. 
    
    Note: This implementation does not support backward propagation, and 
          only supports the 'complete' mode.
    
    See the following regarding this Bug:
        https://github.com/pytorch/pytorch/issues/22573
        https://github.com/cornellius-gp/gpytorch/pull/1224
        
    The input arguments, other than 'gram', follow the PyTorch standard. 
    See the following for their definition:
        https://pytorch.org/docs/stable/generated/torch.linalg.qr.html
        
    Parameters
    ----------
    a: (torch.tensor) the input tensor. Must have a shape of 
        `(*mb_dims, dim, dim)`, where `mb_dims` shows the batch 
        dimensions.

    mode: (str) Either `'complete'` or `'reduced'`. This current 
        implementation only supports the former.
        
    out: (None or torch.tensor) The output tensor for the `Q` matrix. 
        If provided, must have the same shape as `a`.
        
    gram: (str) The Gram-Schmidt process variant. 
    
        * The `classical` variant makes `O(dim)` calls to CUDA 
          and can be more efficient. 
          
        * The `modified` variant can be slightly more accurate, 
          but makes CUDA `O(dim^2)` calls and thus is less efficient.
          
          See Section 14.2 of "Numerical Linear Algebra with Applications" 
          by William Ford on the numerical stability of Gram-Schmidt and 
          its modified variant:
          
          https://www.sciencedirect.com/science/article/abs/pii/B9780123944351000144
          
        * The `cpu` variant uses Pytorch's routine on CPU.
          
        This has to be one of `('classical', 'modified', 'cpu')`.
        
    Output
    ------
    q: (torch.tensor) The output orthonormal matrix. 
        This should have a shape of `(*mb_dims, dim, dim)`.
    
    r: (torch.tensor) The output upper triangle matrix. 
        This should have a shape of `(*mb_dims, dim, dim)`.
    """

    assert not a.requires_grad
    
    if gram == 'cpu':
        with torch.no_grad():
            q, r = torch.linalg.qr(a.detach().cpu(), mode=mode, out=out)
            q_out, r_out = q.to(device=a.device), r.to(device=a.device)
        return q_out, r_out
        
    with torch.no_grad():
        # First Solution: Performing the QR decomposition on CPU
        # Issues: 
        #    1. Pytorch may still only utilize one thread 
        #       practically even though `torch.get_num_threads()` 
        #       may be large.
        #    2. Reliance on CPU resources.
        
        
        ###############################################################
        ################## Initializing & Identifying #################
        ###############################################################
        assert mode == 'complete', 'reduced is not implemented yet'
        # The bactch dimensions
        mb_dims = a.shape[:-2]
        # The input device
        tch_device = a.device
        
        # The Data Type for performing the mathematical caculations
        # Note: Gram-schmidt is numerically unstable. For this reason, even 
        # when the input may be float32, we will do everything in float64.
        tch_dtype = torch.float64
        
        # The QR process dimension
        dim = a.shape[-1]
        assert a.shape == (*mb_dims, dim, dim)

        if out is None:
            q = torch.empty(*mb_dims, dim, dim, device=tch_device, dtype=tch_dtype)
        else:
            q = out
        assert q.shape == (*mb_dims, dim, dim)
        
        # Casting the `a` input to `tch_dtype` and using it from now on
        a_f64 = a.to(dtype=tch_dtype)
        
        ###############################################################
        ################### Performing Gram-Schmidt ###################
        ###############################################################
        if gram == 'classical':
            # Performing the classical Gram-Schmidt Process.
            
            # Creating a copy of `a` to avoid messing up the original input
            acp = a_f64.detach().clone()
            assert acp.shape == (*mb_dims, dim, dim)
            
            for k in range(dim):
                qk_unnorm = acp[..., :, k:k+1]
                assert qk_unnorm.shape == (*mb_dims, dim, 1)

                qk = qk_unnorm / qk_unnorm.norm(dim=-2, keepdim=True)
                assert qk.shape == (*mb_dims, dim, 1)

                a_qkcomps = qk.reshape(*mb_dims, 1, dim).matmul(acp)
                assert a_qkcomps.shape == (*mb_dims, 1, dim)

                # Removing the `qk` components from `a`
                acp -= qk.matmul(a_qkcomps)
                assert acp.shape == (*mb_dims, dim, dim)

                q[..., :, k] = qk.reshape(*mb_dims, dim)
        elif gram == 'modified':
            # Performing the modified Gram-Schmidt Process.
            for i in range(dim):
                q[..., i] = a_f64[..., i]
                for j in range(i):
                    err_ij = torch.einsum('...i,...i->...', q[..., j], q[..., i])
                    assert err_ij.shape == (*mb_dims,)
                    q[..., i] -=  err_ij.reshape(*mb_dims, 1) * q[..., j]
                q[..., i] /= q[..., i].norm(dim=-1, keepdim=True)
        else:
            raise ValueError(f'Unknown gram={gram}')

        r = q.transpose(-1, -2).matmul(a_f64)
        assert r.shape == (*mb_dims, dim, dim)

        ###############################################################
        ##################### Cleanups and Output #####################
        ###############################################################
        # Making sure the lower triangle of `r` is absolutely zero!
        col = torch.arange(dim, device=tch_device, dtype=tch_dtype).reshape(1, dim)
        assert col.shape == (1, dim)

        row = col.reshape(dim, 1)
        assert row.shape == (dim, 1)
        
        mb_ones = [1] * len(mb_dims)
        r *= (row <= col).reshape(*mb_ones, dim, dim)
        
        # Casting the `q` and `r` outputs to the `a` input dtype for compatibility
        q_out, r_out = q.to(dtype=a.dtype), r.to(dtype=a.dtype)
    
    return q_out, r_out

def profmem():
    """
    profiles the memory usage by alive pytorch tensors.
    
    Outputs
    -------
    stats (dict): a torch.device mapping to the number of 
        bytes used by the pytorch tensors.
    """
    fltr_msg  = "torch.distributed.reduce_op is deprecated, "
    fltr_msg += "please use torch.distributed.ReduceOp instead"
    warnings.filterwarnings("ignore", message=fltr_msg)
    fltr_msg  = "`torch.distributed.reduce_op` is deprecated, "
    fltr_msg += "please use `torch.distributed.ReduceOp` instead"
    warnings.filterwarnings("ignore", message=fltr_msg)

    stats = defaultdict(lambda: 0)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') 
                and torch.is_tensor(obj.data)):            
                stats[str(obj.device)] += obj.numel() * obj.element_size()
        except:
            pass
    
    return stats

def batch_kaiming_uniform_(tensor, batch_rng, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    n_seeds, = batch_rng.shape
    assert tensor.shape[0] == n_seeds
    
    fan = _calculate_correct_fan(tensor[0], mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        u = (2 * bound) * batch_rng.uniform(tensor.shape) - bound
        return tensor.copy_(u)


class EMA:
    def __init__(self, gamma, gamma_sq):
        self.gamma = gamma
        self.gamma_sq = gamma_sq
        self.ema = dict()
        self.ema_sq = dict()
    
    def __call__(self, key, val):
        gamma = self.gamma
        gamma_sq = self.gamma_sq
        ema, ema_sq = self.ema, self.ema_sq
        with torch.no_grad():
            val_ema = ema.get(key, val)
            val_ema = gamma * val_ema + (1-gamma) * val
            n_seeds = val_ema.numel()
            ema[key] = val_ema
            
            val_ema_sq = ema_sq.get(key, val**2)
            val_ema_sq = gamma_sq * val_ema_sq + (1-gamma_sq) * (val**2)
            ema_sq[key] = val_ema_sq

            val_popvar = (val_ema_sq - (val_ema**2)).detach().cpu().numpy()
            val_popvar[val_popvar < 0] = 0
            val_ema_std = np.sqrt(val_popvar) * np.sqrt((1-gamma)/(1+gamma))
            
            val_ema_mean = val_ema.mean()
            val_ema_std_mean = np.sqrt((val_ema_std**2).sum()) / n_seeds
        return val_ema_mean, val_ema_std_mean


class BatchRNG:
    is_batch = True

    def __init__(self, shape, lib, device, dtype,
                 unif_cache_cols=1_000_000,
                 norm_cache_cols=5_000_000):
        assert lib in ('torch', 'numpy')

        self.lib = lib
        self.device = device

        self.shape = shape
        self.shape_prod = int(np.prod(self.shape))
        self.shape_len = len(self.shape)
        self.reset_shape_attrs(shape)

        if self.lib == 'torch':
            self.rngs = [torch.Generator(device=self.device)
                         for _ in range(self.shape_prod)]
        else:
            self.rngs = [None for _ in range(self.shape_prod)]
        self.dtype = dtype

        self.unif_cache_cols = unif_cache_cols
        if self.lib == 'torch':
            self.unif_cache = torch.empty((self.shape_prod, self.unif_cache_cols),
                                          device=self.device, dtype=dtype)
        else:
            self.unif_cache = np.empty(
                (self.shape_prod, self.unif_cache_cols), dtype=dtype)
        # So that it would get refilled immediately
        self.unif_cache_col_idx = self.unif_cache_cols
        self.unif_cache_rng_states = None

        self.norm_cache_cols = norm_cache_cols
        if self.lib == 'torch':
            self.norm_cache = torch.empty((self.shape_prod, self.norm_cache_cols),
                                          device=self.device, dtype=dtype)
        else:
            self.norm_cache = np.empty(
                (self.shape_prod, self.norm_cache_cols), dtype=dtype)
        # So that it would get refilled immediately
        self.norm_cache_col_idx = self.norm_cache_cols
        self.norm_cache_rng_states = None

        self.np_qr = np.linalg.qr
        try:
            ver = tuple(int(x) for x in np.__version__.split('.'))
            np_majorver, np_minorver, np_patchver, *_ = ver
            if not ((np_majorver >= 1) and (np_minorver >= 22)):
                def np_qr_fakebatch(a):
                    b = a.reshape(-1, a.shape[-2], a.shape[-1])
                    q, r = list(zip(*[np.linalg.qr(x) for x in b]))
                    qs = np.stack(q, axis=0)
                    rs = np.stack(r, axis=0)
                    qout = qs.reshape(
                        *a.shape[:-2], qs.shape[-2], qs.shape[-1])
                    rout = rs.reshape(
                        *a.shape[:-2], rs.shape[-2], rs.shape[-1])
                    return qout, rout
                self.np_qr = np_qr_fakebatch
        except Exception:
            pass
        
        

    def reset_shape_attrs(self, shape):
        self.shape = shape
        self.shape_prod = int(np.prod(self.shape))
        self.shape_len = len(self.shape)

    def seed(self, seed_arr):
        # Collecting the rng_states after seeding
        assert isinstance(seed_arr, np.ndarray)
        assert len(self.rngs) == seed_arr.size
        flat_seed_arr = seed_arr.copy().reshape(-1)
        if self.lib == 'torch':
            np_random = np.random.RandomState(seed=0)
            for seed, rng in zip(flat_seed_arr, self.rngs):
                np_random.seed(seed)
                balanced_32bit_seed = np_random.randint(
                    0, 2**31-1, dtype=np.int32)
                rng.manual_seed(int(balanced_32bit_seed))
        else:
            self.rngs = [np.random.RandomState(
                seed=seed) for seed in flat_seed_arr]

        if self.unif_cache_col_idx < self.unif_cache_cols:
            self.refill_unif_cache()
            # The cache has been used before, so in order to be able to
            # concat this sampler with the non-reseeded sampler, we should not
            # change the self.unif_cache_cols.

            # Note: We should not refill the uniform cache if the model
            # has not been initialized. This is done to keep the backward
            # compatibility and reproducibility properties with the old scripts.
            # Otherwise, the order of random samplings will change. Remember that
            # the old script first uses dirichlet and priors, and then refills
            # the unif/norm cache. In order to be similar, we should avoid
            # refilling the cache upon the first .seed() call
        if self.norm_cache_col_idx < self.norm_cache_cols:
            self.refill_norm_cache()

    def get_state(self):
        state_dict = dict(unif_cache_rng_states=self.unif_cache_rng_states,
                          norm_cache_rng_states=self.norm_cache_rng_states,
                          norm_cache_col_idx=self.norm_cache_col_idx,
                          unif_cache_col_idx=self.unif_cache_col_idx,
                          rng_states=self.get_rng_states(self.rngs))
        return state_dict

    def set_state(self, state_dict):
        unif_cache_rng_states = state_dict['unif_cache_rng_states']
        norm_cache_rng_states = state_dict['norm_cache_rng_states']
        norm_cache_col_idx = state_dict['norm_cache_col_idx']
        unif_cache_col_idx = state_dict['unif_cache_col_idx']
        rng_states = state_dict['rng_states']

        if unif_cache_rng_states is not None:
            self.set_rng_states(unif_cache_rng_states, self.rngs)
            self.refill_unif_cache()
            self.unif_cache_col_idx = unif_cache_col_idx
        else:
            self.unif_cache_col_idx = self.unif_cache_cols
            self.unif_cache_rng_states = None

        if norm_cache_rng_states is not None:
            self.set_rng_states(norm_cache_rng_states, self.rngs)
            self.refill_norm_cache()
            self.norm_cache_col_idx = norm_cache_col_idx
        else:
            self.norm_cache_col_idx = self.norm_cache_cols
            self.norm_cache_rng_states = None

        self.set_rng_states(rng_states, self.rngs)

    def get_rngs(self):
        return self.rngs

    def set_rngs(self, rngs, shape):
        assert isinstance(rngs, list)
        self.reset_shape_attrs(shape)
        self.rngs = rngs
        assert len(
            self.rngs) == self.shape_prod, f'{len(self.rngs)} != {self.shape_prod}'

    def get_rng_states(self, rngs):
        """
        getting state in ByteTensor
        """
        rng_states = []
        for i, rng in enumerate(rngs):
            rng_state = rng.get_state()
            if self.lib == 'torch':
                rng_state = rng_state.detach().clone()
            rng_states.append(rng_state)
        return rng_states

    def set_rng_states(self, rng_states, rngs):
        """
        rng_states should be ByteTensor (RNG state must be a torch.ByteTensor)
        """
        assert isinstance(
            rng_states, list), f'{type(rng_states)}, {rng_states}'
        for i, rng in enumerate(rngs):
            rs = rng_states[i]
            if self.lib == 'torch':
                rs = rs.cpu()
            rng.set_state(rs)

    def __call__(self, gen, sample_shape):
        assert self.lib == 'torch'
        sample_shape_rightmost = sample_shape[self.shape_len:]
        random_vars = []
        for i, rng in enumerate(self.rngs):
            rng_state = rng.get_state()
            rng_state = rng_state.detach().clone()
            torch.cuda.set_rng_state(rng_state, self.device)
            random_vars.append(gen.sample(sample_shape_rightmost))
            rng.set_state(torch.cuda.get_rng_state(
                self.device).detach().clone())
        rv = torch.stack(random_vars, dim=0).reshape(*sample_shape)
        return rv

    def dirichlet(self, gen_list, sample_shape):
        assert self.lib == 'torch'
        sample_shape_rightmost = sample_shape[self.shape_len:]
        random_vars = []
        for i, (gen_, rng) in enumerate(zip(gen_list, self.rngs)):
            rng_state = rng.get_state().detach().clone()
            torch.cuda.set_rng_state(rng_state, self.device)
            random_vars.append(gen_.sample(sample_shape_rightmost))
            rng.set_state(torch.cuda.get_rng_state(
                self.device).detach().clone())

        rv = torch.stack(random_vars, dim=0)
        rv = rv.reshape(*self.shape, *rv.shape[1:])
        return rv

    def refill_unif_cache(self):
        self.unif_cache_rng_states = self.get_rng_states(self.rngs)
        if self.lib == 'torch':
            for row, rng in enumerate(self.rngs):
                self.unif_cache[row].uniform_(generator=rng)
        else:
            for row, rng in enumerate(self.rngs):
                self.unif_cache[row] = rng.rand(self.unif_cache_cols)

    def refill_norm_cache(self):
        self.norm_cache_rng_states = self.get_rng_states(self.rngs)
        if self.lib == 'torch':
            for row, rng in enumerate(self.rngs):
                self.norm_cache[row].normal_(generator=rng)
        else:
            for row, rng in enumerate(self.rngs):
                self.norm_cache[row] = rng.randn(self.norm_cache_cols)

    def uniform(self, sample_shape):
        sample_shape_tuple = tuple(sample_shape)
        assert sample_shape_tuple[:self.shape_len] == self.shape

        sample_shape_rightmost = sample_shape[self.shape_len:]
        cols = np.prod(sample_shape_rightmost)
        if self.unif_cache_col_idx + cols >= self.unif_cache_cols:
            self.refill_unif_cache()
            self.unif_cache_col_idx = 0

        samples = self.unif_cache[:, self.unif_cache_col_idx: (
            self.unif_cache_col_idx + cols)]
        samples = samples.reshape(*sample_shape)
        self.unif_cache_col_idx += cols

        return samples

    def normal(self, sample_shape):
        sample_shape_tuple = tuple(sample_shape)
        cols = np.prod(sample_shape_tuple) // self.shape_prod
        assert cols * self.shape_prod == np.prod(sample_shape_tuple)
        if self.norm_cache_col_idx + cols >= self.norm_cache_cols:
            self.refill_norm_cache()
            self.norm_cache_col_idx = 0

        samples = self.norm_cache[:, self.norm_cache_col_idx: (
            self.norm_cache_col_idx + cols)]
        samples = samples.reshape(*sample_shape)
        self.norm_cache_col_idx += cols

        return samples

    def so_n(self, sample_shape):        
        sample_shape_tuple = tuple(sample_shape)

        assert sample_shape_tuple[-2] == sample_shape_tuple[-1]
        n_bch, d = self.shape_prod, sample_shape_tuple[-1]
        sample_numel = np.prod(sample_shape_tuple)
        n_v = sample_numel // (self.shape_prod * d * d)
        assert sample_numel == (n_bch * n_v * d * d)
        qr_factorizer = torch_qr_eff if self.lib == 'torch' else self.np_qr
        diagnalizer = torch.diagonal if self.lib == 'torch' else np.diagonal
        signer = torch.sign if self.lib == 'torch' else np.sign

        norms = self.normal((n_bch, n_v, d, d))
        assert norms.shape == (n_bch, n_v, d, d)
        q, r = qr_factorizer(norms)
        assert q.shape == (n_bch, n_v, d, d)
        assert r.shape == (n_bch, n_v, d, d)
        r_diag = diagnalizer(r, 0, -2, -1)
        assert r_diag.shape == (n_bch, n_v, d)
        r_diag_sign = signer(r_diag)
        assert r_diag_sign.shape == (n_bch, n_v, d)
        q_signed = q * r_diag_sign.reshape(n_bch, n_v, 1, d)
        assert q_signed.shape == (n_bch, n_v, d, d)
        so_n = q_signed.reshape(*sample_shape_tuple)
        assert so_n.shape == sample_shape_tuple
        
        return so_n

    @classmethod
    def Merge(cls, sampler1, sampler2):
        assert sampler1.shape_len == sampler2.shape_len == 1

        device = sampler1.device
        dtype = sampler1.dtype
        chain_size = (sampler1.shape[0]+sampler2.shape[0],)

        state_dict1, state_dict2 = sampler1.get_state(), sampler2.get_state()

        merged_state_dict = dict()
        for key in state_dict1:
            if key in ('unif_cache_rng_states', 'norm_cache_rng_states', 'rng_states'):
                # saba modified
                if (state_dict1[key] is None) and (state_dict2[key] is None):
                    merged_state_dict[key] = None
                elif (state_dict1[key] is None) or (state_dict2[key] is None):
                    raise ValueError(f"{key} with None occurance")
                else:
                    merged_state_dict[key] = state_dict1[key] + \
                        state_dict2[key]
            elif key in ('norm_cache_col_idx', 'unif_cache_col_idx'):
                assert state_dict1[key] == state_dict2[key]
                merged_state_dict[key] = state_dict1[key]
            else:
                raise ValueError(f'Unknown rule for {key}')

        sampler = cls(device, chain_size, dtype)
        sampler.set_state(merged_state_dict)
        return sampler

    @classmethod
    def Subset(cls, sampler, inds):
        assert sampler.shape_len == 1

        device = sampler.device
        dtype = sampler.dtype
        chain_size_sub = (len(inds),)

        state_dict = sampler.get_state()

        sub_state_dict = dict()
        for key in state_dict:
            if key in ('unif_cache_rng_states', 'norm_cache_rng_states',
                       'rng_states'):
                sub_state_dict[key] = [state_dict[key][ind] for ind in inds]
            elif key in ('norm_cache_col_idx', 'unif_cache_col_idx'):
                sub_state_dict[key] = state_dict[key]
            else:
                raise ValueError(f'Unknown rule for {key}')

        sampler = cls(device, chain_size_sub, dtype)
        sampler.set_state(sub_state_dict)
        return sampler


class bffnn(nn.Module):
    """batched FF network for approximating functions"""

    def __init__(self, inp_width, nn_width, num_hidden, activation, shape, batch_rng, out_width=1):
        super().__init__()
        act_dict = dict(silu=nn.SiLU(), tanh=nn.Tanh(), relu=nn.ReLU())
        self.layer_first = nn.ParameterList(
            self.make_linear(shape, inp_width, nn_width, batch_rng))
        layers_w, layers_b = [], []
        for _ in range(num_hidden):
            w, b = self.make_linear(shape, nn_width, nn_width, batch_rng)
            layers_w.append(w)
            layers_b.append(b)
        self.layer_hidden_w = nn.ParameterList(layers_w)
        self.layer_hidden_b = nn.ParameterList(layers_b)
        self.layer_last = nn.ParameterList(
            self.make_linear(shape, nn_width, out_width, batch_rng))
        self.inp_width = inp_width
        self.out_width = out_width

        self.shape = shape
        self.ndim = len(shape)
        self.size = int(np.prod(shape))
        self.activation = act_dict[activation.lower()]

    def forward(self, x):
        activation = self.activation
        assert x.shape[:self.ndim] == self.shape
        assert x.shape[-1] == self.inp_width
        bdims = self.shape
        hidden_w, hidden_b = self.layer_hidden_w, self.layer_hidden_b
        inp_width = self.inp_width
        out_width = self.out_width
        x_middims = x.shape[self.ndim:-1]
        x_pts = int(np.prod(x_middims))
        # x.shape --> ( n_bch, n_srf,  n_points,        d)

        u = x.reshape(*bdims,                x_pts, inp_width)
        # u.shape --> (n_bch, n_srf * n_points,         d)

        w, b = self.layer_first
        # w.shape --> (n_bch,                    d,   nn_width)
        # b.shape --> (n_bch,                    1,   nn_width)

        u = activation(torch.matmul(u, w) + b)
        # u.shape --> (n_bch, n_srf * n_points,   nn_width)

        for _, (w, b) in enumerate(zip(hidden_w, hidden_b)):
            u = activation(torch.matmul(u, w) + b)
        # u.shape --> (n_bch, n_srf * n_points,   nn_width)

        w, b = self.layer_last
        u = torch.matmul(u, w) + b
        # u.shape --> (n_bch, n_srf * n_points,          1)

        u = u.reshape(*x.shape[:-1], out_width)
        # u.shape --> (n_bch, n_srf,  n_points,          1)
        return u

    def make_linear(self, shape, inp_width, out_width, batch_rng):
        k = 1. / np.sqrt(inp_width).item()
        with torch.no_grad():
            w_unit = batch_rng.uniform((*shape, inp_width, out_width))
            b_unit = batch_rng.uniform((*shape,         1, out_width))
            w_tensor = w_unit * (2 * k) - k
            b_tensor = b_unit * (2 * k) - k
        w = torch.nn.Parameter(w_tensor)
        b = torch.nn.Parameter(b_tensor)
        return w, b


class bcnn2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                 kernel_size, activation, do_batchnorm, shape, batch_rng,
                 stride=1, padding=0, dilation=1, 
                 output_padding=0, in_height=None, in_width=None,
                 transpose_conv=False,  transpose_x=True):
        super().__init__()
        n_seeds, = shape
        tch_device = batch_rng.device
        tch_dtype = batch_rng.dtype
        act_dict = dict(silu=nn.SiLU(), tanh=nn.Tanh(), relu=nn.ReLU(), 
            lrelu=nn.LeakyReLU(), eye=nn.Identity())
        
        x_inshape = (in_channels, in_height, in_width)
        #########################################################
        ################ Creating the CNN Layers ################
        #########################################################
        def int2pair(x):
            o = [x, x] if isinstance(x, int) else x
            assert isinstance(o, (list, tuple))
            assert len(o) == 2
            return o
        
        kernel_size = int2pair(kernel_size)
        stride = int2pair(stride)
        padding = int2pair(padding)
        dilation = int2pair(dilation)
        output_padding = int2pair(output_padding)
        
        cnn_layers, x_shapes, bn_layers = [], [], []
        h_width, h_height = in_width, in_height
        h_channels = in_channels
        for h_channels_out in hidden_channels:
            if not transpose_conv:
                cnn_layer = nn.Conv2d(in_channels=n_seeds*h_channels, 
                    out_channels=n_seeds*h_channels_out, kernel_size=kernel_size, 
                    stride=stride, padding=padding, groups=n_seeds,
                    dilation=dilation, device=tch_device, dtype=tch_dtype)
            else:
                cnn_layer = nn.ConvTranspose2d(in_channels=n_seeds*h_channels, 
                    out_channels=n_seeds*h_channels_out, kernel_size=kernel_size, 
                    stride=stride, padding=padding, groups=n_seeds,
                    dilation=dilation, output_padding=output_padding, 
                    device=tch_device, dtype=tch_dtype)
            cnn_layers.append(cnn_layer)
            
            bn_layer = None
            if do_batchnorm:
                bn_layer = nn.BatchNorm2d(n_seeds*h_channels_out, 
                    device=tch_device, dtype=tch_dtype)
            bn_layers.append(bn_layer)
            
            if h_height is not None:
                if not transpose_conv:
                    h_height_ = h_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
                    h_height = int((h_height_ / stride[0]) + 1)
                else:
                    h_height  = ((h_height - 1) * stride[0] - 2 * padding[0] +
                                dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1)
            
            if h_width is not None:
                if not transpose_conv:
                    h_width_ = h_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
                    h_width = int((h_width_ / stride[1]) + 1)
                else:
                    h_width = ((h_width - 1) * stride[1] - 2 * padding[1] +
                            dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1)
                
            h_channels = h_channels_out
            x_shapes.append((h_channels, h_height, h_width))
        
        #########################################################
        ############# Storing the Class Variables ###############
        #########################################################
        self.shape = (n_seeds,)
        self.n_seeds = n_seeds
        self.device = tch_device
        self.dtype = tch_dtype
        self.batch_rng = batch_rng
        self.transpose_conv = transpose_conv
        self.transpose_x = transpose_x
        self.activation = act_dict[activation]
        
        self.cnn_layers = nn.ParameterList(cnn_layers)
        self.bn_layers = bn_layers if not(do_batchnorm) else nn.ParameterList(bn_layers)
        self.x_inshape = x_inshape
        self.x_shapes = x_shapes
        self.kernel_size = kernel_size
        
        # Initializing the layers
        self.init_layers()
            
    def forward(self, x):
        n_seeds = self.n_seeds
        h_channels, h_height, h_width = self.x_inshape
        
        h_height_ = h_height if h_height is not None else x.shape[-2]
        h_width_ = h_width if h_width is not None else x.shape[-1]
        
        if self.transpose_x:
            n_samps = x.shape[1]
            assert x.shape == (n_seeds, n_samps, h_channels, h_height_, h_width_)
            x = x.transpose(0, 1)
        else:
            n_samps = x.shape[0]
        assert x.shape == (n_samps, n_seeds, h_channels, h_height_, h_width_)

        x = x.reshape(n_samps, n_seeds * h_channels, h_height_, h_width_)
        assert x.shape == (n_samps, n_seeds * h_channels, h_height_, h_width_)
        
        for cnn_layer, bn_layer, x_shape in zip(self.cnn_layers, self.bn_layers, self.x_shapes):
            x = cnn_layer(x)
            h_channels, h_height, h_width = x_shape
            h_height_ = h_height if h_height is not None else x.shape[-2]
            h_width_ = h_width if h_width is not None else x.shape[-1]
            
            assert x.shape == (n_samps, n_seeds * h_channels, h_height_, h_width_)
            
            if bn_layer is not None:
                x = bn_layer(x)
                assert x.shape == (n_samps, n_seeds * h_channels, h_height_, h_width_)
            
            x = self.activation(x)
            assert x.shape == (n_samps, n_seeds * h_channels, h_height_, h_width_)
         
        x = x.reshape(n_samps, n_seeds, h_channels, h_height_, h_width_)
        assert x.shape == (n_samps, n_seeds, h_channels, h_height_, h_width_)
        
        if self.transpose_x:
            x = x.transpose(0, 1)
            assert x.shape == (n_seeds, n_samps, h_channels, h_height_, h_width_)
        
        return x
    
    def init_layers(self):
        k0, k1 = self.kernel_size
        n_seeds = self.n_seeds
        h_channels, h_height, h_width = self.x_inshape
        batch_rng = self.batch_rng
        transpose_conv = self.transpose_conv
        
        with torch.no_grad():
            #########################################################
            ############## Initializing the CNN Layers ##############
            #########################################################
            for layer, x_shape in zip(self.cnn_layers, self.x_shapes):
                h_channels_out, h_height, h_width = x_shape
                
                if not transpose_conv:
                    h1, h2 = h_channels_out, h_channels
                else:
                    h1, h2 = h_channels, h_channels_out
                
                w = layer.weight
                assert w.shape == (n_seeds * h1, h2, k0, k1)
                w2 = w.reshape(n_seeds, h1, h2, k0, k1)
                assert w2.shape == (n_seeds, h1, h2, k0, k1)
                w2 = batch_kaiming_uniform_(w2, batch_rng, a=math.sqrt(5))
                assert w2.shape == (n_seeds, h1, h2, k0, k1)
                
                b = layer.bias
                if b is not None:
                    assert b.shape == (n_seeds * h_channels_out,)
                    fan_in, _ = _calculate_fan_in_and_fan_out(w2[0])
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        u1 = batch_rng.uniform((n_seeds, h_channels_out))
                        assert u1.shape == (n_seeds, h_channels_out)
                        u2 = (2 * bound) * u1.reshape(n_seeds * h_channels_out) - bound
                        assert u2.shape == (n_seeds * h_channels_out,)
                        assert u2.shape == b.shape
                        b.copy_(u2)
                        
                h_channels = h_channels_out

if __name__ == '__main__':
    ###############################################################
    ################# Unit-testing `torch_qr_eff` #################
    ###############################################################
    n_bch = 1000000
    dim = 10
    
    torch.manual_seed(12345)
    tch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tch_dtype = torch.float64
    a = torch.randn(n_bch, dim, dim, device=tch_device, dtype=tch_dtype)

    q, r = torch_qr_eff(a, mode='complete', gram='classical')

    rtol = 1e-05
    atol = 1e-06

    # Test 1: Checking if `q` is orthonormal
    eye = torch.eye(dim, device=tch_device, dtype=tch_dtype).reshape(1, dim, dim)
    assert q.transpose(-1, -2).matmul(q).allclose(eye, rtol=rtol, atol=atol)

    # Test 2: Checking if `a == q @ r` holds
    assert a.allclose(q.matmul(r), rtol=rtol, atol=atol)

    # Test 3: Checking if `r` is upper-triangle
    col = torch.arange(dim, device=tch_device, dtype=tch_dtype
        ).reshape(1, 1, dim).expand(n_bch, 1, dim)
    assert col.shape == (n_bch, 1, dim)
    row = col.reshape(n_bch, dim, 1)
    assert row.shape == (n_bch, dim, 1)
    r_lowtriang = r[row > col]
    assert r_lowtriang.allclose(torch.zeros_like(r_lowtriang), rtol=rtol, atol=atol)
