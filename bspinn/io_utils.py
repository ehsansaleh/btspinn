# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3.8
#     language: python
#     name: p38
# ---

# ## I/O Utilities for Pre-processing the Config Dictionaries

import numpy as np
from collections import OrderedDict as odict
from itertools import product, chain
from collections import ChainMap
from textwrap import dedent
import subprocess
import h5py
import fnmatch
import time
from os.path import exists, getmtime
import pandas as pd
import math

import io
import os
import glob
from collections import defaultdict
from pandas.api.types import union_categoricals
from bspinn.io_cfg import results_dir, nullstr


# + code_folding=[3, 112, 209, 239, 826]
#########################################################
############ Pre-processing JSON Config Files ###########
#########################################################
class Looper:
    """
    A Looper object that recurively creates looper children for
    itself if it gets a string child. The `opt2vals` input will be
    mutated and emptied out after the `pop_vals` calls, so make sure
    you take a backup copy for yourself before giving it to this
    constructor.

    Parameters
    ----------
    childern: (list of str or Looper): a list of either option names
        (as string) or child Looper objects.

        Example: childern = ['lr', OvatLooper(...), 'tau',
                             'problem', ZipLooper(...),
                             ...]

    opt2vals: (dict) a config dictionary with keys mapped to
        a list of looped values. All options (even the ones
        that are fixed and don't need looping over) must be
        mapped to a list (albeit a single item list).

        Example: opt2vals = {'lr': [0.001],
                             'problem': ['poisson'],
                             'tau': [0.999, 0.9999, 0.999999],
                             ...}
    """

    def __init__(self, children, opt2vals):
        self.opt2vals = opt2vals
        self.children = []
        self.patterns = []
        for child in children:
            if isinstance(child, Looper):
                self.children.append(child)
                self.patterns.append(None)
            elif isinstance(child, str):
                pat = child
                self.patterns.append(pat)
                if pat == "rest":
                    child_ = AtomicLooper(None, opt2vals)
                    self.children.append(child_)
                else:
                    opts = fnmatch.filter(opt2vals.keys(), pat)
                    msg_ = f'Did not find "{child}" for looping: '
                    msg_ += f"{list(opt2vals.keys())}"
                    assert len(opts) >= 1, msg_
                    child_ = AtomicLooper(opts, opt2vals)
                    self.children.append(child_)
        self.lstrip_prfxs = []

    def pop_vals(self, turn):
        """
        Recursively calls pop_vals on its children until it reaches all
        leaf (i.e., AtomicLooper) objects.
        """
        for child in self.children:
            child.pop_vals(turn)

    def lstrip(self, *lstrip_prfxs):
        """
        Memorizes a bunch of lstrip prefixes so that it applies the
        stripping at the end of the get_cfgs() calls (i.e., during the
        inner pack call). Returns self.
        """
        self.lstrip_prfxs += list(lstrip_prfxs)
        return self

    def pack(self, cfgs):
        """
        This function does two things:
          1. Wraps the input in another list. This is needed for
             singleton loopers such as (cart, ovat, zip), which merge
             all the children dictionaries togather.

          2. Applies the lstrip function with the stored prefixes on each
             of the dictionary keys.

        Parameters
        ----------
        cfgs: (list of dict) a list of dictionaries
            Example:
                cfgs = [{'a': 1,  'b': 4,  'c': 7},
                        {'a': 2,  'b': 5,  'c': 8},
                        {'a': 3,  'b': 6,  'c': 9},
                        {'d': 10, 'e': 13},
                        {'d': 11, 'e': 14},
                        {'d': 12, 'e': 15}]

        Output
        ------
        out: (list of list of dict) a list-wrapped version of the input.
            Example:
                cfgs = [[{'a': 1,  'b': 4,  'c': 7},
                         {'a': 2,  'b': 5,  'c': 8},
                         {'a': 3,  'b': 6,  'c': 9},
                         {'d': 10, 'e': 13},
                         {'d': 11, 'e': 14},
                         {'d': 12, 'e': 15}]]
        """
        if len(self.lstrip_prfxs) == 0:
            out = [cfgs]
        else:
            out = [
                [
                    odict(
                        (lstrip(k, self.lstrip_prfxs), v)
                        for k, v in cfg.items()
                    )
                    for cfg in cfgs
                ]
            ]
        return out

    def get_cfgs(self):
        raise NotImplementedError("This should have been overriden!")


class AtomicLooper:
    """
    A class resembling the Looper object with a slight difference
    that its instances will be placed at the leaves of the looping
    tree. This does not make any recursive calls to its children
    (since it has no children).

    Parameters
    ----------
    optnames: (list of str): a list of option names.

        Example: optnames = ['lr', 'tau', 'problem']

    opt2vals: (dict) a config dictionary with keys mapped to
        a list of looped values. All options (even the ones
        that are fixed and don't need looping over) must be
        mapped to a list (albeit a single item list).

        Example: opt2vals = {'lr': [0.001],
                             'problem': ['poisson'],
                             'tau': [0.99, 0.999, 0.9999],
                             ...}
    """

    def __init__(self, optnames, opt2vals):
        self.opt2vals = opt2vals
        self.optnames = optnames
        self.values = None
        self.is_rest = optnames is None
        self.cfg = None

    def pop_vals(self, turn):
        """
        Transfers the options and their values from the globally shared
        `self.opt2vals` dictionary to its own `self.cfg` dictionary.
        """
        if turn == "nonrest":
            if not self.is_rest:
                cfg = odict()
                for opt in self.optnames:
                    msg_ = f'Option "{opt}" was either (a) not defined '
                    msg_ += "in the original config, or (b) two loopers"
                    msg_ += " attempted to use it."
                    assert opt in self.opt2vals, msg_
                    cfg[opt] = self.opt2vals.pop(opt)
                self.cfg = cfg
        elif turn == "rest":
            if self.is_rest:
                rmng_opts = list(self.opt2vals.keys())
                cfg = odict()
                for opt in rmng_opts:
                    cfg[opt] = self.opt2vals.pop(opt)
                msg_ = 'A "rest" option turned out to be empty in the '
                msg_ += 'looping tree. Either \n\t(1) "rest" was used '
                msg_ += "more than once in the looping tree, or \n\t(2) "
                msg_ += "all options were already specified in the looping "
                msg_ += 'tree and nothing was left for "rest" to claim.'
                assert len(cfg) > 0, msg_
                self.cfg = cfg
        else:
            raise ValueError(f"turn {turn} undefined.")

    def get_cfgs(self, pass_list=False):
        """
        The main function applying the looping process and
        generating dictionaries.

        Example 1:
            pass_list = False

            self.cfg = {'a': [1, 2],
                        'b': [3, 4],
                        'c': [5, 6]}

            out = [[{'a': 1}, {'a': 2}],
                   [{'b': 3}, {'b': 4}],
                   [{'c': 5}, {'c': 6}]]

        Example 2:
            pass_list = True

            self.cfg = {'a': [1, 2],
                        'b': [3, 4],
                        'c': [5, 6]}

            out = [[{'a/list': [1, 2]},
                   [{'b/list': [3, 4]},
                   [{'c/list': [5, 6]}]]
        """
        if not pass_list:
            out = [
                [{opt: val} for val in vals] for opt, vals in self.cfg.items()
            ]
        else:
            out = [[{f"{opt}/list": vals}] for opt, vals in self.cfg.items()]
        return out


class CartLooper(Looper):
    def get_cfgs(self):
        """
        A few examples will clarify the data structure:

        Example:
            child_cfglists = [[[{'a': 1}, {'a': 2}],
                               [{'b': 3}, {'b': 4}],
                               [{'c': 5}, {'c': 6}]],
                              [[{'d': 7}, {'d': 8}]],
                              [[{'e': 9}, {'e': 0}]]]

            child_cfgs = [[{'a': 1}, {'a': 2}],
                          [{'b': 3}, {'b': 4}],
                          [{'c': 5}, {'c': 6}],
                          [{'d': 7}, {'d': 8}],
                          [{'e': 9}, {'e': 0}]]

            cfgs = [{'a':1, 'b':3, 'c':5, 'd':7, 'e':9},
                    {'a':1, 'b':3, 'c':5, 'd':7, 'e':0},
                    {'a':1, 'b':3, 'c':5, 'd':8, 'e':9},
                    ...]

            output = [cfgs]
        """
        child_cfglists = [child.get_cfgs() for child in self.children]
        child_cfgs = chain.from_iterable(child_cfglists)
        cfgs = list(chain_dicts(dtup) for dtup in product(*child_cfgs))
        return self.pack(cfgs)


class ZipLooper(Looper):
    def get_cfgs(self):
        """
        The main function applying the looping process and
        generating the output config dictionaries. Recursive
        calls of this method in the children function will
        build up the user config dictionaries.

        A few examples will clarify the data structure:

        Example:
            child_cfglists = [[[{'a': 1}, {'a': 2}],
                            [{'b': 3}, {'b': 4}],
                            [{'c': 5}, {'c': 6}]],
                            [[{'d': 7}, {'d': 8}]],
                            [[{'e': 9}, {'e': 0}]]]

            child_cfgs = [[{'a': 1}, {'a': 2}],
                          [{'b': 3}, {'b': 4}],
                          [{'c': 5}, {'c': 6}],
                          [{'d': 7}, {'d': 8}],
                          [{'e': 9}, {'e': 0}]]

            cfgs = [{'a':1, 'b':3, 'c':5, 'd':7, 'e':9},
                    {'a':2, 'b':4, 'c':6, 'd':8, 'e':0}]

            output = [cfgs]
        """
        child_cfglists = [child.get_cfgs() for child in self.children]
        child_cfgs = chain.from_iterable(child_cfglists)
        cfgs = list(chain_dicts(dtup) for dtup in zip(*child_cfgs))
        return self.pack(cfgs)


class OvatLooper(Looper):
    def get_cfgs(self):
        """
        The main function applying the looping process and
        generating the output config dictionaries. Recursive
        calls of this method in the children function will
        build up the user config dictionaries.

        A few examples will clarify the data structure:

        Example:
            child_cfglists = [[[{'a': 1}, {'a': 2}],
                            [{'b': 3}, {'b': 4}],
                            [{'c': 5}, {'c': 6}]],
                            [[{'d': 7}, {'d': 8}]],
                            [[{'e': 9}, {'e': 0}]]]

            child_cfgs = [[{'a': 1}, {'a': 2}],
                        [{'b': 3}, {'b': 4}],
                        [{'c': 5}, {'c': 6}],
                        [{'d': 7}, {'d': 8}],
                        [{'e': 9}, {'e': 0}]]

            cfgs = [{'a':1, 'b':3, 'c':5, 'd':7, 'e':9},
                    {'a':2, 'b':3, 'c':5, 'd':7, 'e':9},
                    {'a':1, 'b':4, 'c':5, 'd':7, 'e':9},
                    {'a':1, 'b':3, 'c':6, 'd':7, 'e':9},
                    {'a':1, 'b':3, 'c':5, 'd':8, 'e':9},
                    {'a':1, 'b':3, 'c':5, 'd':7, 'e':0}]

            output = [cfgs]
        """
        child_cfglists = [child.get_cfgs() for child in self.children]
        child_cfgs = chain.from_iterable(child_cfglists)
        cfgs = list(chain_dicts(dtup) for dtup in ovat(*child_cfgs))
        return self.pack(cfgs)


class AsListLooper(Looper):
    def get_cfgs(self):
        """
        The main function applying the looping process and
        generating the output config dictionaries. Recursive
        calls of this method in the children function will
        build up the user config dictionaries.

        A few examples will clarify the data structure:

        Example:
            child_cfglists = [[[{'a': 1}],
                            [{'b': 3}],
                            [{'c': 5}]],
                            [[{'d': 7}]],
                            [[{'e': 9}]]]

            child_cfgs = [[{'a': 1}],
                        [{'b': 3}],
                        [{'c': 5}],
                        [{'d': 7}],
                        [{'e': 9}]]

            cfgd_lst = [{'a': 1},
                        {'b': 3},
                        {'c': 5},
                        {'d': 7},
                        {'e': 9}]

            cfgs = [{'a':1, 'b':3, 'c':5, 'd':7, 'e':9}]

            output = [cfgs]
        """
        msg_ = "aslist only accepts options and cant work "
        msg_ += "with loopers such as cart/zip/..."
        assert all(
            isinstance(child, AtomicLooper) for child in self.children
        ), msg_

        child_cfglists = [
            child.get_cfgs(pass_list=True) for child in self.children
        ]
        child_cfgs = list(chain.from_iterable(child_cfglists))
        assert all(len(lst) == 1 for lst in child_cfgs)
        cfgd_lst = [lst[0] for lst in child_cfgs]
        cfgs = [chain_dicts(cfgd_lst)]
        return self.pack(cfgs)


class CatLooper(Looper):
    def get_cfgs(self):
        """
        The main function applying the looping process and
        generating the output config dictionaries. Recursive
        calls of this method in the children function will
        build up the user config dictionaries.

        A few examples will clarify the data structure:

        Example 1:
            child_cfglists_ = [[[{'a': 1,  'b': 4,  'c': 7},
                                {'a': 2,  'b': 5,  'c': 8},
                                {'a': 3,  'b': 6,  'c': 9}],
                            [[{'d': 10, 'e': 13},
                                {'d': 11, 'e': 14},
                                {'d': 12, 'e': 15}]]]

            child_cfglists = child_cfglists_

            child_cfgs = [[{'a': 1,  'b': 4,  'c': 7},
                        {'a': 2,  'b': 5,  'c': 8},
                        {'a': 3,  'b': 6,  'c': 9}],
                        [{'d': 10, 'e': 13},
                        {'d': 11, 'e': 14},
                        {'d': 12, 'e': 15}]]

            cfgs = [{'a': 1,  'b': 4,  'c': 7},
                    {'a': 2,  'b': 5,  'c': 8},
                    {'a': 3,  'b': 6,  'c': 9},
                    {'d': 10, 'e': 13},
                    {'d': 11, 'e': 14},
                    {'d': 12, 'e': 15}]


        Example 2 (curating singletons):
            child_cfglists_ = [[[{'a': 1}],
                                [{'b': 4}],
                                [{'c': 7}]]
                            [[{'d': 10, 'e': 13},
                                {'d': 11, 'e': 14},
                                {'d': 12, 'e': 15}]]]

            child_cfglists = [[[{'a': 1, 'b': 4, 'c': 7}]]
                            [[{'d': 10, 'e': 13},
                                {'d': 11, 'e': 14},
                                {'d': 12, 'e': 15}]]]

            child_cfgs = [[{'a': 1,  'b': 4,  'c': 7}],
                        [{'d': 10, 'e': 13},
                        {'d': 11, 'e': 14},
                        {'d': 12, 'e': 15}]]

            cfgs = [{'a': 1,  'b': 4,  'c': 7},
                    {'d': 10, 'e': 13},
                    {'d': 11, 'e': 14},
                    {'d': 12, 'e': 15}]

            output = [cfgs]
        """
        child_cfglists_ = [child.get_cfgs() for child in self.children]

        # If the `child_cfglists` are uncrated, yet each one of them is
        # a singleton, it doesn't matter if you combine them using cart
        # or zip or ovat. For this, we just combine them ourselves.
        # See Example 2 for a demonstration.
        child_cfglists = []
        for i, cfglst in enumerate(child_cfglists_):
            all_singltons = all(len(lst) == 1 for lst in cfglst)
            if all_singltons:
                merged_lst = [chain_dicts([lst[0] for lst in cfglst])]
                child_cfglists.append([merged_lst])
            else:
                child_cfglists.append(cfglst)

        # Making sure no uncurated options are left
        msg_ = "You seem to have passed a bunch of uncurated options "
        msg_ += "to the cat looper. Make sure you merge each group of "
        msg_ += "cat options using one of the merger loopers (e.g., "
        msg_ += "cart, zip, or ovat) before passing it to cat.\n"
        msg_ += f"    child_cfglists = {child_cfglists}"
        assert all(len(cfglst) == 1 for cfglst in child_cfglists), msg_

        child_cfgs = chain.from_iterable(child_cfglists)
        cfgs = list(chain.from_iterable(child_cfgs))
        return self.pack(cfgs)


def ovat(*lists):
    """
    Gets a bunch of lists and loops over them in a
    "one variable at a time" manner.

    Parameters
    ----------
    lists (list of list): a list of lists.

        Example:
            lists = [[{'a': 1}, {'a': 2}],
                     [{'b': 3}, {'b': 4}],
                     [{'c': 5}, {'c': 6}],
                     [{'d': 7}, {'d': 8}],
                     [{'e': 9}, {'e': 0}]]

    Output
    ------
    out: (list of list) the ovat loopeings of the input.

        Example:
            out = [[{'a': 1}, {'b': 3}, {'c': 5}, {'d': 7}, {'e': 9}],
                   [{'a': 2}, {'b': 3}, {'c': 5}, {'d': 7}, {'e': 9}],
                   [{'a': 1}, {'b': 4}, {'c': 5}, {'d': 7}, {'e': 9}],
                   [{'a': 1}, {'b': 3}, {'c': 6}, {'d': 7}, {'e': 9}],
                   [{'a': 1}, {'b': 3}, {'c': 5}, {'d': 8}, {'e': 9}],
                   [{'a': 1}, {'b': 3}, {'c': 5}, {'d': 7}, {'e': 0}]]
    """
    assert all(isinstance(ll, list) for ll in lists)
    cntrdval = tuple(vv[0] for vv in lists)
    othvals = [
        (*cntrdval[:vv_idx], v, *cntrdval[(vv_idx + 1):])
        for vv_idx, vv in enumerate(lists)
        for v in vv[1:]
    ]
    out = [cntrdval] + othvals
    return out


def chain_dicts(dicts_iterable):
    """
    Chains (i.e., comibnes) a bunch of dictionaries into a single
    ordered dictionary.

    Parameters
    ----------
    dicts_iterable: (iterable of dict) a list of dictionaries that need
        to be combined together.

        Example:
            dicts_iterable = [{'a': 1}, {'b': 2}, {'c': 3}]

    Output
    ------
    out: (odict) a single combined dictionary.

        Example:
            out = {'a': 1, 'b': 2, 'c': 3}
    """
    return odict(ChainMap(*dicts_iterable[::-1]))


def lstrip(s, prfx_lst):
    """
    Left-strips a bunch of prefixes from a string. If there
    is more than a single prefix, the order of prefixes for
    stripping must be chosen carefully (see the example).

    Parameters
    ----------
    s: (str) the input string to be lstripped.

        Example: s = 'g0/g1/g3/g2/opt'

    prfx_lst: (list of str) a list of prefixes for
        left-stripping in an ordered fashion.

        Example: prfx_lst = ['g0/', 'g1/', 'g2/', 'g3/']

    Output
    ------
    out: (str) the left stripped string

        Example: out = 'g3/g2/opt'
    """
    out = s
    for prfx in prfx_lst:
        out = out.lstrip(prfx)
    return out


def looper_maker(looping_str, opt2vals):
    """
    This function takes a string python command (which is defined in a
    json config file), and then executes it using the eval function in
    python so that it produces an Looper object.

    Parameters
    ----------
    looping_str: (str) the python command describing the looping process.

        Example 1: "cart('all')"

        Example 2: "cart('dim', 'n_srf', 'n_epochs', 'lr')"

        Example 3: "cart('dim', zip('n_epochs', ovat('n_srf', 'lr')))"

    Outputs
    ----------
    looper: (Looper) An object created recursively by the eval function.
    """
    def cart(*args):
        return CartLooper(args, opt2vals)

    def ovat(*args):
        return OvatLooper(args, opt2vals)

    def zip(*args):
        return ZipLooper(args, opt2vals)

    def cat(*args):
        return CatLooper(args, opt2vals)

    def aslist(*args):
        return AsListLooper(args, opt2vals)

    _ = (cart, ovat, zip, cat, aslist)

    looper = eval(looping_str)

    return looper


def cfg_iters2list(cfg_dict):
    """
    Converting "range" or "linspace" specifications into a list of values.

    Parameters
    ----------
    cfg_dict: (dict) the config dictionary coming from a json file.

        Example: {"desc/lines": ["hello", "world!"],
                  "rng_seeds/range": [0, 5000, 1000],
                  "dim/list": [2, 3, 4],
                  "n_vol/linspace": [400, 100, 4],
                  "n_epochs/loglinspace": [200, 25, 4],
                  "lr/logrange": [0.001, 0.000001, 0.1],
                  "tau": 0.999}

    Outputs
    ----------
    proced_cfg_dict: (dict) the same dictionary with all keys having a
        "/list", "/range", "/logrange", "/linspace", or "/loglinspace"
        postfix removed and treated. The values will now be a list
        even if they originally were not (see "tau" for example).

        Example: {"desc": ["hello\nworld!"],
                  "rng_seeds": [0, 1000, 2000, 3000, 4000],
                  "dim": [2, 3, 4],
                  "n_vol": [400, 300, 200, 100],
                  "n_epochs": [200, 100, 50, 25],
                  "lr": [0.001, 0.0001, 0.00001],
                  "tau": [0.999]}
    """
    proced_cfg_dict = odict()
    for key, val in cfg_dict.items():
        if key.endswith("/range") or key.endswith("/logrange"):
            assert len(val) == 3
            need_logtrans = key.endswith("/logrange")
            proced_key = "/".join(key.split("/")[:-1])
            st, end, step = val
            if need_logtrans:
                st, end = math.log(st), math.log(end)
            proced_val_arr = np.arange(st, end, step)
            if need_logtrans:
                proced_val_arr = proced_val_arr.exp()
            caster = int if all(isinstance(v, int) for v in val) else float
        elif key.endswith("/linspace") or key.endswith("/loglinspace"):
            assert len(val) == 3
            need_logtrans = key.endswith("/loglinspace")
            proced_key = "/".join(key.split("/")[:-1])
            st, end, num = val
            if need_logtrans:
                st, end = math.log(st), math.log(end)
            proced_val_arr = np.linspace(st, end, num)
            if need_logtrans:
                proced_val_arr = proced_val_arr.exp()
            caster = int if all(isinstance(v, int) for v in val) else float
        elif key.endswith("/lines"):
            msg_ = f'The value for "{key}" option is not a list of strings: '
            msg_ += 'Options ending in "/lines" must specify '
            msg_ += "a list of strings."
            assert isinstance(val, list), msg_
            assert all(isinstance(vln, str) for vln in val), msg_
            proced_key = key[:-6]
            proced_val_arr = ["\n".join(val)]
            caster = None
        elif key.endswith("/list"):
            assert isinstance(val, list), f'"{key}" should specify a list!'
            proced_key = key[:-5]
            proced_val_arr = val
            caster = None
        else:
            proced_key, proced_val_arr, caster = key, [val], None

        if caster is not None:
            proced_val = [caster(v) for v in proced_val_arr]
            assert all(
                np.all(v1 == v2) for v1, v2 in zip(proced_val, proced_val_arr)
            )
        else:
            proced_val = proced_val_arr

        msg_ = f'"{proced_key}" was specified in multiple ways! '
        msg_ += 'Remember that for an option "X", only one of '
        msg_ += '"X" or "X/list" or "X/range" or "X/logrange" '
        msg_ += 'or "X/linspace" or "X/loglinspace" or "X/lines"'
        msg_ += " should be specified."
        assert proced_key not in proced_cfg_dict, msg_
        proced_cfg_dict[proced_key] = proced_val
    return proced_cfg_dict


def preproc_cfgdict(cfg_dict, opt_order="config"):
    """
    This function pre-process a config dict read from a json
    file, and returns a list of config dicts after applying the
    desired looping strategy.

    Parameters
    ----------
    cfg_dict: (dict) config dict read from a json file.

        Example: {
          "desc": "Example scratch json config",
          "date": "December 20, 2022",
          "problem": "poisson",
          "rng_seed/range": [0, 5000, 1000],
          "dim/list": [2, 3, 4],
          "n_srf/list": [400, 200, 100],
          "n_srfpts_mdl": 0,
          "n_srfpts_trg": 2,
          "trg/w": 1.0,
          "n_epoch/list": [200000, 100000],
          "do_detspacing": false,
          "do_bootstrap": false,
          "do_dblsampling": true,
          "lr/list": [0.001, 0.0001],
          "tau": 0.999,
          "looping": "cart(aslist('rng_seed'), zip('dim', 'lr'), 'rest')"
        }

    opt_order: (str) determines the order of the options in the
        key should be either 'config' or 'looping'.

    Outputs
    ----------
    all_cfgdicts: (list of dicts) list of processed config dicts.

        Example: [{'desc': 'Example scratch json config',
                   'date': 'December 20, 2022',
                   'problem': 'poisson',
                   'rng_seed/list': [0, 1000, 2000, 3000, 4000],
                   'dim': 2,
                   'n_srf': 400,
                   'n_srfpts_mdl': 0,
                   'n_srfpts_trg': 2,
                   'trg/w': 1.0,
                   'n_epoch': 200000,
                   'do_detspacing': False,
                   'do_bootstrap': False,
                   'do_dblsampling': True,
                   'lr': 0.001,
                   'tau': 0.999},
                    ...]
    """

    # Removing "/list", "/range", "/linspace", etc. and
    # replacing them with a proper list of hp values.
    opt2vals_orig = cfg_iters2list(cfg_dict)

    # `opt_order='config'` will try to preserve the order of options
    # in the output dictionary as the json config file. Otherwise, the
    # order will follow the DFS order of options in the looping tree.
    # `opt_order='looping'` will make the options follow the DFS order
    # of the looping tree.
    assert opt_order in ("config", "looping")

    if "looping" in opt2vals_orig:
        opt2vals_ = opt2vals_orig.copy()
        n_opts = len(opt2vals_orig)
        looping_str = opt2vals_.pop("looping")[0]

        # The looper orderings are the same as to the looping tree.
        # The trick to preserve the config order in the options, is
        # to store the index of options along with the values, and
        # then sorting the keys at the end using those indexes.
        if opt_order == "config":
            # Example:
            #     opt2vals_ = {'a': [10, 20, 30],
            #                  'b': [40, 50],
            #                  'c': [60]}
            #
            #     opt2vals  = {'a': [(0, 10), (0, 20), (0, 30)],
            #                  'b': [(1, 40), (1, 50)],
            #                  'c': [(2, 60)]}
            opt2vals = odict()
            for optidx, (opt, vals) in enumerate(opt2vals_.items()):
                opt2vals[opt] = [(optidx, val) for val in vals]
        elif opt_order == "looping":
            opt2vals = opt2vals_.copy()
        else:
            raise ValueError(f"Unknown opt_order={opt_order}")

        # Obtaining the looper object using python's `eval()` method
        # on the looping string defined in the json config file.
        looper = looper_maker(looping_str, opt2vals)

        # First, we need to ask all non-'rest' looping leaf objects
        # (i.e., AtomicLooper) to claim their values, and pop their
        # options out of the `opt2vals` dictionary
        looper.pop_vals(turn="nonrest")

        # Next, we ask the 'rest' looping leaf objects to claim any
        # options that are left for themselves. There should only be
        # a single 'rest' looper. If there are more than one, the
        # second one will be empty of options.
        looper.pop_vals(turn="rest")

        # Now we should make sure that 'opt2vals' is emptied out and
        # all options were somehow used in the looping process.
        msg_ = f"""\
        The following options were left unspecified in the
        looping tree. Make sure you include all options in
        the looping tree. As a hint, you can use the "rest"
        key to insert the remaining options:
            {list(opt2vals.keys())}
        """
        assert len(opt2vals) == 0, dedent(msg_)

        # The main function call: Getting the list of all looped
        # config dictionaries.
        rt_cfgs = looper.get_cfgs()

        # Making sure nothing ambiguous is left to deal with.
        msg_rt = "The looping tree seems ambiguous and some of the "
        msg_rt += "options still seem uncurated. As a hint for solving "
        msg_rt += "this, you should put one of singleton loopers (e.g., "
        msg_rt += "cart, zip, or ovat) at the root of the looping tree!"
        assert len(rt_cfgs) == 1, msg_rt
        all_cfgs = rt_cfgs[0]

        # Taking care of the option sorting after the looper generated
        # looping-sorted dictionaries :)
        if opt_order == "config":
            all_srtdcfgs = []
            for cfg in all_cfgs:
                srtdcfg = odict()

                # Since the 'aslist' looper keeps the '*/list' option values
                # intact, the option indecis remain distributed among the
                # values. We should refactorize the index values before going
                # any further.
                def cure_aslist(opt, idx_vals):
                    if opt.endswith("/list"):
                        # Example:
                        #   opt = 'a/list'
                        #   idx_vals = [(0, 10), (0, 20), (0, 30)]
                        #   out = (0, [10, 20, 30])
                        opt_idxs = [x[0] for x in idx_vals]
                        assert len(set(opt_idxs)) <= 1
                        vals = [x[1] for x in idx_vals]
                        if len(opt_idxs) > 0:
                            opt_idx = opt_idxs[0]
                        else:
                            opt_idx = n_opts
                        out = (opt_idx, vals)
                    else:
                        out = idx_vals

                    # Some sanity checks
                    assert isinstance(out, tuple)
                    assert len(out) == 2
                    assert isinstance(out[0], int)

                    return out

                cfg = odict(
                    [
                        (opt, cure_aslist(opt, idx_vals))
                        for opt, idx_vals in cfg.items()
                    ]
                )

                # Applying the sorting process
                def srt_key(opt):
                    return cfg[opt][0]

                for opt in sorted(cfg, key=srt_key):
                    srtdcfg[opt] = cfg[opt][1]
                all_srtdcfgs.append(srtdcfg)
            all_cfgs = all_srtdcfgs
    else:
        # No modifications are necessary if the `'looping'` key is
        # absent from the config dictionary.
        all_cfgs = [cfg_dict]

    return all_cfgs


#########################################################
################### Generic Utilties ####################
#########################################################


def get_git_commit():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode()
        )
    except Exception:
        commit_hash = "no git repo found"
    return commit_hash


#########################################################
################### Data Writer Class ###################
#########################################################


class DataWriter:
    def __init__(self, flush_period=10, compression_level=9):
        self.flush_period = flush_period
        self.file_path = None
        self.key2cdi = odict()
        self.gnrlidx = 0
        self.compression_level = compression_level

    def set_path(self, file_path):
        if self.file_path is not None:
            if file_path != self.file_path:
                print(
                    f"DW: need to flush after {self.gnrlidx} rows "
                    "due to a set_path call",
                    flush=True,
                )
                self.flush()
                self.gnrlidx = 0
        self.file_path = file_path

    @property
    def data_len(self):
        dlen = 0
        if len(self.key2cdi) > 0:
            dlen = max(
                len(index)
                for key, (colt_t, data, index) in self.key2cdi.items()
            )
        return dlen

    @property
    def file_ext(self):
        assert self.file_path is not None
        if self.file_path.endswith(".csv"):
            return "csv"
        elif self.file_path.endswith(".h5"):
            return "h5"
        else:
            raise ValueError(f"Unknown extension for {self.file_path}")

    def add(self, data_tups, file_path):
        if (file_path is None) or (len(data_tups) == 0):
            return

        self.set_path(file_path)

        # Making sure all options in all hdf keys have the same
        # length (i.e., number of rows)
        opt2len = {
            f"{key}/{opt}": len(vals)
            for key, rows_dict, col_type in data_tups
            for opt, vals in rows_dict.items()
        }
        msg_ = "The number of rows must be the same across"
        msg_ += f" all keys and options:\n{opt2len}"
        assert len(set(opt2len.values())) == 1, msg_
        n_rows = list(opt2len.values())[0]
        newidxs = list(range(self.gnrlidx, self.gnrlidx + n_rows))

        # Checking if shapes have changed and therefore flush if needed.
        need_flush = False
        for key, rows_dict, col_t_ in data_tups:
            if key in self.key2cdi:
                col_t, data, index = self.key2cdi[key]
                if (col_t == "np.arr") and (self.file_ext == "h5"):
                    for i, (opt, val) in enumerate(rows_dict.items()):
                        assert isinstance(val, np.ndarray)
                        if opt in data:
                            a, b = val, data[opt][0]
                            issubd = np.issubdtype(a.dtype, b.dtype)
                            if (a.shape[1:] != b.shape[1:]) or not (issubd):
                                need_flush = True

        if need_flush:
            print(
                f"DW: need to flush after {self.gnrlidx} rows "
                "due to shape mismatch.",
                flush=True,
            )
            self.flush(forget=True)

        # Inserting the new data
        for key, rows_dict, col_t_ in data_tups:
            if key not in self.key2cdi:
                col_t, data, index = col_t_, odict(), []
                self.key2cdi[key] = (col_t, data, index)
            else:
                col_t, data, index = self.key2cdi[key]
                assert col_t == col_t_, f"{key} was {col_t} not {col_t_}"

                msg_ = f"{key} input opts and my columns are different:\n"
                msg_ += f"  input opts: {set(rows_dict.keys())}\n"
                msg_ += f"  my columns: {set(data.keys())}\n"
                assert list(data.keys()) == list(rows_dict.keys()), msg_

            assert col_t in ("np.arr", "pd.cat", "pd.qnt")
            assert (self.file_ext != "csv") or (col_t in ("pd.cat", "pd.qnt"))

            for i, (opt, val) in enumerate(rows_dict.items()):
                if opt not in data:
                    data[opt] = []
                if col_t == "np.arr":
                    val_lst = [val]
                else:
                    val_lst = list(val)
                data[opt] += val_lst
            index += newidxs

        self.gnrlidx += n_rows
        if (self.data_len >= self.flush_period) and (self.flush_period > 0):
            print(
                f"DW: need to flush after {self.gnrlidx} "
                "rows due to io/flush frequency.",
                flush=True,
            )
            self.flush(forget=True)

    def flush(self, forget=True):
        if self.data_len == 0:
            return None
        assert self.file_path is not None

        flsh_sttime = time.time()
        if self.file_ext == "csv":
            for key, (col_t, data, index) in self.key2cdi.items():
                assert col_t in ("pd.cat", "pd.qnt"), "np.arr not impl. yet"
                pddtype = "category" if col_t == "pd.cat" else None
                data_df = pd.DataFrame(data, dtype=pddtype)
                data_df["ioidx"] = index
                columns = list(data_df.columns)
                csvfpath = self.file_path.replace(
                    ".csv", f'_{key.replace("/", "_")}.csv'
                )
                # Appending the latest row to file_path
                if not os.path.exists(csvfpath):
                    data_df.to_csv(
                        csvfpath,
                        mode="w",
                        header=True,
                        index=False,
                        columns=columns,
                    )
                else:
                    # First, check if we have the same columns
                    old_cols = pd.read_csv(csvfpath, nrows=1).columns.tolist()
                    old_cols_set = set(old_cols)
                    my_cols_set = set(columns)
                    msg_assert = "file columns and my columns are different:\n"
                    msg_assert = msg_assert + f"  file cols: {old_cols_set}\n"
                    msg_assert = msg_assert + f"  my columns: {my_cols_set}\n"
                    assert old_cols_set == my_cols_set, msg_assert
                    data_df.to_csv(
                        csvfpath,
                        mode="a",
                        header=False,
                        index=False,
                        columns=old_cols,
                    )
        elif self.file_ext == "h5":
            assert all(
                col_t in ("pd.cat", "pd.qnt", "np.arr")
                for key, (col_t, data, index) in self.key2cdi.items()
            )
            ##################################################################
            ##################### Pandas HDF Operations ######################
            ##################################################################
            # Getting the next part index to write
            nextpart = 0
            if exists(self.file_path):
                h5hdf = h5py.File(self.file_path, mode="r", driver="core")
                roots = set(x for x in h5hdf.keys() if x.startswith("P"))
                roots = set(x[1:] for x in roots)
                roots = set(x for x in roots if x.isdigit())
                roots = set(int(x) for x in roots)
                nextpart = max(roots) + 1 if len(roots) > 0 else 0
                assert f"{nextpart:08d}" not in h5hdf.keys()
                h5hdf.close()

            # Writing the pandas dataframes
            pdhdf = pd.HDFStore(self.file_path, mode="a")
            for key, (colt_t, data, index) in self.key2cdi.items():
                if colt_t in ("pd.cat", "pd.qnt"):
                    # Writing the main table
                    pddtype = "category" if colt_t == "pd.cat" else None
                    pdfmt = "table" if colt_t == "pd.cat" else None
                    data_df = pd.DataFrame(data, dtype=pddtype)
                    bools_d = {
                        col: "bool"
                        for col, v_lst in data.items()
                        if all(isinstance(v, bool) for v in v_lst)
                    }
                    if len(bools_d) > 0:
                        data_df = data_df.astype(bools_d)
                    data_df.to_hdf(
                        pdhdf,
                        key=f"/P{nextpart:08d}/{key}",
                        format=pdfmt,
                        index=False,
                        append=False,
                        complib="zlib",
                        complevel=self.compression_level,
                    )
            pdhdf.close()

            # Writing the numpy array datasets
            h5hdf = h5py.File(self.file_path, mode="a", driver="core")
            for key, (colt_t, data, index) in self.key2cdi.items():
                np_idx = np.array(index)
                h5hdf.create_dataset(
                    f"/P{nextpart:08d}/{key}/ioidx",
                    shape=np_idx.shape,
                    dtype=np_idx.dtype,
                    data=np_idx,
                    compression="gzip",
                    compression_opts=9,
                )
                if colt_t in ("np.arr",):
                    for opt, np_arr_list in data.items():
                        np_arr = np.concatenate(np_arr_list, axis=0)
                        h5hdf.create_dataset(
                            f"/P{nextpart:08d}/{key}/{opt}",
                            shape=np_arr.shape,
                            dtype=np_arr.dtype,
                            data=np_arr,
                            compression="gzip",
                            compression_opts=self.compression_level,
                        )
            h5hdf.close()
        else:
            raise ValueError(
                f"file extension not implemented for {self.file_path}."
            )

        self.key2cdi = odict()
        print(f"DW: flush took {time.time()-flsh_sttime:.3f} seconds.")

    def close(self):
        if self.file_path is not None:
            msg_ = f"DW: need to flush after {self.gnrlidx}"
            msg_ += " rows due to close call."
            print(msg_, flush=True)
        return self.flush(forget=True)


#########################################################
######## Hierarchical-Deep Dictionary Conversion ########
#########################################################


def hie2deep(hie_dict, dictcls=None, maxdepth=None, sep="/"):
    """
    Gets a hierarchical dictionary and returns a deep
    dictionary conversion of it. Raises an error if a
    bare key is present along with its children (e.g.,
    {'a': 1.0, 'a/b': 2.0} will raise an error).

    Parameters
    ----------
    hie_dict: (dict) a hierarchical dictionary
        Example: {'a/b/c': 1.,
                  'a/b/d': 2.}

    dictcls: (class or None) the dict constructor

    maxdepth: (int) the maximum depth until which the
        conversion recursion can go. If None, no max depth
        constraint will be applied.

    sep: (str) the directory seperator character, such as
         '/' in unix addressing.

    Output
    ----------
    deep_dict: (dict) a deep dictionary
        Example: {'a': {'b': {'c': 1., 'd': 2.}}}
    """
    if dictcls is None:
        dictcls = odict if isinstance(hie_dict, odict) else dict

    lvl1keys = list(odict.fromkeys(key.split(sep)[0] for key in hie_dict))
    chmaxdepth = None if maxdepth is None else (maxdepth - 1)
    cangodeeper = (chmaxdepth is None) or (chmaxdepth > 0)
    deep_dict = dictcls()
    for key in lvl1keys:
        subdict = dictcls()
        for k, v in hie_dict.items():
            prfx = f"{key}{sep}"
            if k.startswith(prfx):
                subdict[k[len(prfx):]] = v

        if len(subdict) > 0:
            msg_ = f'cannot have "{key}" when its children '
            msg_ += f"are present: {hie_dict}"
            assert key not in hie_dict, msg_
            if cangodeeper:
                deep_dict[key] = hie2deep(subdict, dictcls, chmaxdepth, sep)
            else:
                deep_dict[key] = subdict
        else:
            deep_dict[key] = hie_dict[key]
    return deep_dict


def deep2hie(deep_dict, dictcls=None, maxdepth=None, sep="/"):
    """
    Gets a deep dictionary and returns a hierarchical
    dictionary conversion of it.

    Parameters
    ----------
    deep_dict: (dict) a deep dictionary
        Example: {'a': {'b': {'c': 1., 'd': 2.}}}


    dictcls: (class or None) the dict constructor

    maxdepth: (int) the maximum depth until which the
        conversion recursion can go. If None, no max depth
        constraint will be applied.

    sep: (str) the directory seperator character, such as
         '/' in unix addressing.

    Output
    ----------
    hie_dict: (dict) a hierarchical dictionary
        Example: {'a/b/c': 1.,
                  'a/b/d': 2.}
    """
    if dictcls is None:
        dictcls = odict if isinstance(deep_dict, odict) else dict

    chmaxdepth = None if maxdepth is None else (maxdepth - 1)
    cangodeeper = (chmaxdepth is None) or (chmaxdepth > 0)

    hie_dict = dictcls()
    for key, val in deep_dict.items():
        if isinstance(val, dict) and cangodeeper:
            subdict = deep2hie(val, dictcls, chmaxdepth, sep)
            for k, v in subdict.items():
                hie_dict[f"{key}{sep}{k}"] = v
        else:
            hie_dict[key] = val

    return hie_dict


#########################################################
########### Sanity Checking Utility Functions ###########
#########################################################


msg_bcast = "{} should be np broadcastable to {}={}. "
msg_bcast += "However, it has an inferred shape of {}."


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
    msg_ = f'"{name}" must be in trns_opts but it isnt: {trns_opts}'
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

    is_bcastble = all(
        (x == y or x == 1 or y == 1) for x, y in zip(src_shape, trg_shape)
    )
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
    opt_dstr = cfgdict.get(f"{opt}/dstr", "fixed")

    msg_ = f"Unknown {opt}_dstr: it should be one of {list(dstr2args.keys())}"
    assert opt_dstr in dstr2args, msg_

    opt2req = dict() if opt2req is None else opt2req
    optreqs = opt2req.get(opt, tuple())
    must_spec = list(dstr2args[opt_dstr]) + list(optreqs)
    avid_spec = list(
        chain.from_iterable(v for k, v in dstr2args.items() if k != opt_dstr)
    )
    avid_spec = [k for k in avid_spec if k not in must_spec]

    if opt_dstr == "fixed":
        # To avoid infinite recursive calls, we should end this here.
        msg_ = f'"{opt}" must be specified.'
        if parnt_optdstr is not None:
            parnt_opt, parnt_dstr = parnt_optdstr
            msg_ += f'"{parnt_opt}" was specified as "{parnt_dstr}", and'
        msg_ += f' "{opt}" was specified as "{opt_dstr}".'
        if len(optreqs) > 0:
            msg_ += f'Also, "{opt}" requires "{optreqs}" to be specified.'
        opt_val = cfgdict.get(opt, None)
        assert opt_val is not None, msg_
    else:
        for arg in must_spec:
            opt_arg = f"{opt}{arg}"
            chck_dstrargs(opt_arg, cfgdict, dstr2args,
                          opt2req, (opt, opt_dstr))

    for arg in avid_spec:
        opt_arg = f"{opt}{arg}"
        opt_arg_val = cfgdict.get(opt_arg, None)
        msg_ = f'"{opt_arg}" should not be specified, since "{opt}" '
        msg_ += f'appears to follow the "{opt_dstr}" distribution.'
        assert opt_arg_val is None, msg_


# -

#####################################################
########## Dataframe Manipulation Utilties ##########
#####################################################


def drop_unqcols(df):
    """
    Returns a copy of the dataframe where all columns
    contaning a single unique value are dropped.

    Parameters
    -----------
    df: (pd.DataFrame) the dataframe with redundant
        columns.
    """
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df = df.drop(columns=col)
    return df


def get_dfidxs(df, cdict):
    """
    Generates a boolean index array for the rows of the `df`
    dataframe. Each row will be ture iff it has identical values
    to the ones in `cdict`.

    Parameters
    ----------
    df: (pd.DataFrame) the main dataframe.
        Example:
            d = {'col1': [ 0,   1,   2,   3 ],
                 'col2': ['a', 'b', 'a', 'd']}
            df = pd.DataFrame(d)

    cdict: (dict) a dictionary of conditioning values
        for the indexing.
        Example: cdict = {'col2': 'a'}

    Output:
    -------
    idx: (np.array) a boolean numpy array indicating whether
        each row of `df` has the same values as `cdict`.

        Example: idx = np.array([0, 2])
    """
    idx = np.full((df.shape[0],), True)
    for col, val in cdict.items():
        idx = np.logical_and(idx, (df[col] == val).values)
    return idx


def get_ovatgrps(hpdf):
    """
    Takes a hyper-parameters data-frame and divides the
    `fpidx` values into groups. This function assumes an
    ovat-style expriment (i.e., one variable at a time).
    The central hyper-parameters must be located at the
    top of the data-frame.

    Parameters
    ----------
    hpdf (pd.DataFrame) a data-frame only containing the
        hyper-parameters of the experiments. Having duplicate
        rows is fine, but do not
    """
    uhpdf = hpdf.drop_duplicates().copy()
    uhpdf = uhpdf.drop("fpidx", axis=1)
    uhpdf = drop_unqcols(uhpdf)
    uhpdf = uhpdf.reset_index(drop=True)

    mainrow = uhpdf.iloc[0]
    diffgrps = defaultdict(list)
    for i, row in uhpdf.iterrows():
        aa = row.eq(mainrow)
        diffs = {
            bb: row[bb]
            for bb, cc in dict(aa).items()
            if (not (cc) and (row[bb] != nullstr))
        }
        fpidx = row["fpidxgrp"]
        if len(diffs) == 0:
            continue
        diffs["fpidxgrp"] = fpidx
        diffgrps[tuple(sorted(tuple(diffs.keys())))].append(diffs)

    fpgrps = []
    for dgi, (dgkeys, dglist) in enumerate(diffgrps.items()):
        # dglist is a list of diff dictionaries
        dglist.append({key: mainrow[key] for key in dgkeys})
        dglist = sorted(dglist, key=lambda diffs: diffs[dgkeys[0]])
        dgdf = pd.DataFrame(dglist).drop_duplicates()
        if dgdf.shape[1] > 1:
            sortcols = list(dgdf.columns)
            sortcols.remove("fpidxgrp")
            dgdf = dgdf[["fpidxgrp"] + sortcols]
            dgdf = dgdf.sort_values(by=sortcols)
        fpgrps.append(dgdf["fpidxgrp"].tolist())

    # Example:
    #   fpgrps = [['01_poisson/02_mse2d.0.0',
    #              '01_poisson/02_mse2d.3.0',
    #              '01_poisson/02_mse2d.5.0',
    #              '01_poisson/02_mse2d.0.1'],
    #             ['01_poisson/02_mse2d.2.2',
    #              '01_poisson/02_mse2d.2.1',
    #              '01_poisson/02_mse2d.0.0'],
    #             ['01_poisson/02_mse2d.0.3',
    #              '01_poisson/02_mse2d.0.0']]
    return fpgrps


def pd_concat(dfs, *args, **kwargs):
    """
    Concatenate while preserving categorical columns.

    Parameters
    ----------
    dfs: (list of pd.DataFrame) a list of data-frames
        with categorical columns to be concatenated.

    *args: arguments to be piped to `pd.concat`.

    **kwargs: keyword arguments to be piped to `pd.concat`.

    Output
    ------
    catdf: (pd.DataFrame) a concatenated pandas dataframe.
    """
    # Iterate on categorical columns common to all dfs
    dfs = [df.copy(deep=False) for df in dfs]
    for col in set.intersection(
        *[set(df.select_dtypes(include="category").columns) for df in dfs]
    ):
        # Generate the union category across dfs for this column
        uc = union_categoricals([df[col] for df in dfs])
        # Change to union category for all dataframes
        for df in dfs:
            df[col] = pd.Categorical(df[col].values, categories=uc.categories)
    catdf = pd.concat(dfs, *args, **kwargs)
    return catdf


def to_pklhdf(
    df,
    path,
    key,
    mode="a",
    driver="core",
    compression="gzip",
    compression_opts=0,
):
    """
    Writes the dataframe into a byte-stream on the RAM, and
    then transfers it into a numpy array inside the hdf file.

    This function was written due to exhausting number of errors
    associated with saving categorical pandas dataframes into hdf
    files directly.

    Parameters
    ----------
    df: (pd.DataFrame) a pandas data-frame either with categorical
        or non-categorical data.

    path: (str) the path of the hdf file on the disk

    key: (str) the hdf key to write the df into

    mode: (str) piped to `h5py.File`

    driver: (str) piped to `h5py.File`

    compression: (str) piped to `hdf.create_dataset`

    compression_opts: (int) piped to `hdf.create_dataset`
    """
    stream = io.BytesIO()
    df.to_pickle(stream)
    np_arr = np.frombuffer(stream.getbuffer(), dtype=np.uint8)
    with h5py.File(path, mode=mode, driver=driver) as h5hdf:
        h5hdf.create_dataset(
            key + "/pkl",
            shape=np_arr.shape,
            dtype=np_arr.dtype,
            data=np_arr,
            compression=compression,
            compression_opts=compression_opts,
        )


def read_pklhdf(path, key, driver="core"):
    """
    Reads the pickled numpy array from the hdf file into RAM,
    transforms it into a bytesarray, and then reads it with
    pd.read_pkl.

    This function was written due to exhausting number of errors
    associated with saving categorical pandas dataframes into hdf
    files directly.

    Parameters
    ----------
    path: (str or h5py.File or h5py.Group or h5py.Dataset) the path
        of the hdf file on the disk. It can be an already opened file
        pointer using the h5py library as well.

    key: (str) the hdf key to write the df into

    driver: (str) piped to `h5py.File`

    Output
    ------
    df: (pd.DataFrame) a pandas data-frame either with categorical
        or non-categorical data.
    """
    open_fp = isinstance(path, str)
    if open_fp:
        h5hdf = h5py.File(path, mode="r", driver=driver)
    else:
        h5hdf = path
    rstream = io.BytesIO(h5hdf[key + "/pkl"][:].tobytes())
    if open_fp:
        h5hdf.close()
    df = pd.read_pickle(rstream)
    return df


#####################################################
########### HDF File Manipulation Utilties ##########
#####################################################


def get_ioidxkeys(h5fp):
    """
    Gets all keys that in the h5 file/group that end with the
    '/ioidx' string.

    Parameters
    ----------
    h5fp: (h5py.File or h5py.Group) a file pointer opened with
        the h5py library, or an h5py group handle.

    Outputs
    -------
    all_ioidxs: (list of str) a list of keys, where `f{key}/ioidx`
        exists in the `h5fp` group.
    """
    all_ioidxs = []

    def list_ioidxs(name, obj):
        if name.endswith("/ioidx"):
            all_ioidxs.append(name[:-6])

    h5fp.visititems(list_ioidxs)
    return all_ioidxs


def get_ioidx(h5fp, key):
    """
    Gets the `ioidx` array for a particular key in the hdf file.
    It traverses all its parents, until it finds the first grandparent
    that has an `ioidx` child.

    Parameters
    ----------
    h5fp: (h5py.File or h5py.Group) a file pointer opened with
        the h5py library, or an h5py group handle.

    key: (str) a key for which the ioidx must be found.
        Example: key = 'var/eval/ug/sol/mdl'

    Outputs
    -------
    ioidx: (np.array) a numpy array which is the `ioidx` of key.
    """
    hirar = key.split("/")
    n_hirar = len(hirar)
    possioilocs = [
        "/".join(hirar[: n_hirar - i] + ["ioidx"]) for i in range(n_hirar)
    ]
    ioidxpath = max(possioilocs[::-1], key=lambda loc: loc in h5fp)
    ioidx = h5fp[ioidxpath][:]
    return ioidx


def get_h5leafs(h5fp):
    """
    Gets all leafs in the h5 file/group. Leafs are array datasets
    that do not have any further sub-directories.

    Parameters
    ----------
    h5fp: (h5py.File or h5py.Group) a file pointer opened with
        the h5py library, or an h5py group handle.

    Outputs
    -------
    leafs: (list of str) a list of dataset keys.
    """
    if isinstance(h5fp, h5py.Dataset):
        leafs = [""]
    else:
        leafs = []

        def addl(name, obj):
            if isinstance(obj, h5py.Dataset):
                leafs.append(name)

        h5fp.visititems(addl)
    return leafs


def get_h5dtypes(h5grp):
    """
    Queies all the keys inside the hdf file and returns a
    dictionary specifying them and their respective dtypes.

    The output will be a dictionary mapping each key to its
    dtype.  The dtype values are one of 'np.arr', 'pd.cat',
    or 'pd.qnt'.

    The output will only include keys corresponding to h5py
    datasets or pandas dataframes. In other words, h5py groups
    (i.e., directories) will not be included in the output.

    Parameters
    ----------
    h5grp: (h5py.File or h5py.Group) a file pointer opened with
        the h5py library, or an h5py group handle.

    Outputs
    -------
    key2dtype: (dict) a key-to-dtype mapping of the
        existing keys to their data types.
        Example:

        key2dtype = {
            'etc': 'pd.cat',
            'hp': 'pd.cat',
            'stat': 'pd.qnt',
            'mon': 'pd.qnt',
            'var/eval/ug/sol/mdl': 'np.arr',
            'var/eval/ug/sol/gt': 'np.arr',
            'mdl/layer_first.0': 'np.arr',
            ...
        }

    """
    aaa = get_h5leafs(h5grp)
    bbb = [x for x in aaa if not x.endswith("/ioidx")]
    pdpklkeys = set(x.split("/pkl")[0] for x in bbb if "/pkl" in x)
    pdcatkeys = set(x.split("/meta")[0] for x in bbb if "/meta" in x)
    pdqntkeys = set(
        x.split("/axis")[0]
        for x in bbb
        if ("/axis" in x) and (x not in pdcatkeys)
    )
    dfkeys = pdcatkeys.union(pdqntkeys).union(pdpklkeys)
    npkeys = [x for x in bbb if not any(x.startswith(y) for y in dfkeys)]
    key2dtype = {
        **{x: "pd.cat" for x in pdcatkeys},
        **{x: "pd.pkl" for x in pdpklkeys},
        **{x: "pd.qnt" for x in pdqntkeys},
        **{x: "np.arr" for x in npkeys},
    }
    return key2dtype


def get_h5keys(h5grp):
    return list(get_h5dtypes(h5grp).keys())


def save_h5data(data, path, driver='core'):
    """
    Saves a dictionary with pands/array values into an hdf file.

    Parameters
    ----------
    data: (dict) a dictionary mapping hdf keys to their values.
        The values could either be pandas dataframes or numpy
        arrays.

    path: (str) the path of the hdf file.
    """
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            dtype = "np.arr"
        elif isinstance(val, pd.DataFrame):
            is_cat = any(
                val[col].dtype.name == "category" for col in val.columns
            )
            dtype = "pd.cat" if is_cat else "pd.qnt"
        else:
            dtype = None
            raise ValueError(f"{type(val)} is not acceptable")

        if dtype == "pd.cat":
            to_pklhdf(val, path, key, mode="a")
        elif dtype == "pd.qnt":
            val.to_hdf(
                path,
                key=key,
                mode="a",
                format=None,
                index=False,
                append=False,
                complib="zlib",
                complevel=0,
            )
        elif dtype == "np.arr":
            with h5py.File(path, mode="a", driver=driver) as h5hdf:
                h5hdf.create_dataset(
                    key,
                    shape=val.shape,
                    dtype=val.dtype,
                    data=val,
                    compression="gzip",
                    compression_opts=0,
                )
        else:
            raise ValueError(f"{dtype} not defined")


def load_h5data(path, driver="core"):
    """
    Loads the hdf data into a dictionary

    Parameters
    ----------
    path: (str) the path of the hdf file to be read

    Output
    ------
    data: (dict) a dictionary mapping hdf keys to their values.
        The values could either be pandas dataframes or numpy
        arrays.
    """
    h5fp = h5py.File(path, mode="r", driver=driver)
    grpdtypes = get_h5dtypes(h5fp)
    data = dict()
    for key, dtype in grpdtypes.items():
        if dtype in ("pd.cat", "pd.qnt"):
            data[key] = pd.read_hdf(path, key=key)
        elif dtype in ("pd.pkl",):
            data[key] = read_pklhdf(h5fp, key)
        elif dtype in ("np.arr",):
            data[key] = h5fp[key][:]
        else:
            raise ValueError(f"dtype={dtype} not defined")
    h5fp.close()
    return data


def print_duinfo(duinfo):
    tot_nbyes = 0
    for key, nbytes in duinfo.items():
        a = key + " " * (77 - len(key))
        b = f"{nbytes/1e6:.3f}"
        b = " " * (10 - len(b)) + b
        print(f"{a}{b} MB")
        tot_nbyes += nbytes
    print("-" * 90)
    c = f"{tot_nbyes/1e9:.3f}"
    c = " " * (10 - len(c)) + c
    print("Total" + " " * 72 + f"{c} GB", flush=True)


def get_h5du(h5fp, verbose=False, detailed=True, driver="core"):
    """
    Computes the disk usage of all the child datasets
    under a h5 group.

    Parameters
    ----------
    h5fp: (str or h5py.File or h5py.Group) a file pointer opened
        with the h5py library, or an h5py group handle. If a string
        is provided, the function will treat it as a path and open
        it in read mode.

    verbose: (bool) whether to print the disk usage information in
        a human readable table.

    detailed: (bool) whether to open up the panads/pytables inner keys
        or hide them in a single dataframe sum.

    Outputs
    -------
    key2nbytes: (dict) a dictionary mapping each child dataset to
        its disk usage in bytes.
    """

    needs_open = isinstance(h5fp, str)
    if needs_open:
        if verbose:
            print(f'Disk usage for "{h5fp}"')
        h5grp = h5py.File(h5fp, mode="r", driver=driver)
    else:
        h5grp = h5fp

    leafs = get_h5leafs(h5grp)
    h5dss = [h5grp if k == "" else h5grp[k] for k in leafs]
    h5dss = [(k, h5grp if k == "" else h5grp[k]) for k in leafs]
    key2nbytes = {k: ds.attrs.get("nelements", ds.nbytes) for k, ds in h5dss}

    if not detailed:
        key2nbytes = {
            key: sum(v for k, v in key2nbytes.items() if k.startswith(key))
            for key in get_h5keys(h5grp)
        }

    if verbose:
        print_duinfo(key2nbytes)

    if needs_open:
        h5grp.close()

    return key2nbytes


def rslv_fpidx(fpidx, h5_opener=None):
    """
    Resolves an `fpidx` string into a list of resolved
    `fpidx` strings.

    A resolved `fpidx` must be formatted as
        f'{cfg_tree}.{fidx}.{pidx}',
    where
        1. `cfg_tree` is the config tree,
        Example:  cfg_tree = '01_poisson/02_mse2d'

        2. `fidx` is the file index, and should be
        an integer. For example, `fidx=5` points to
        the '01_poisson/02_mse2d_05.h5' hdf file.

        3. `pidx` is an int refering to the partition
        number inside the hdf file.
        For example, 'pidx=3' refers to the
        `/P00000003/` group inside the hdf file.

    Unresolved `fpidx` strings are ones which either
        1. do not specify the file and partition index, or
        2. specify one or both of them with a '*'.

    If `fpidx` is already resolved, it will be included in
    the output as-is.

    Parameters
    ----------
    fpidx: (str) an unresolved `fpidx` string
        Example:
        fpidx = '01_poisson/02_mse2d'

        fpidx = '01_poisson/02_mse2d.*.*'

        fpidx = '01_poisson/02_mse2d.0.*'

        fpidx = '01_poisson/02_mse2d.*.0'

    h5_opener: (callable) a function that takes the `cfg_tree`
        and `fidx` arguments and returns an h5py.File object.
        The `fidx` argument is an integer, and `cfg_tree` is
        a string.

    fpidx_list: (list of str) a list of resolved `fpidx`s.
        Example:
        fpidx = ['01_poisson/02_mse2d.0.0',
                    '01_poisson/02_mse2d.1.0',
                    ...,
                    '01_poisson/02_mse2d.5.8']

        fpidx = ['01_poisson/02_mse2d.0.0',
                    '01_poisson/02_mse2d.1.0',
                    ...,
                    '01_poisson/02_mse2d.5.8']

        fpidx = ['01_poisson/02_mse2d.0.0',
                    '01_poisson/02_mse2d.0.1',
                    ...,
                    '01_poisson/02_mse2d.0.8']

        fpidx = ['01_poisson/02_mse2d.0.0',
                    '01_poisson/02_mse2d.1.0',
                    ...,
                    '01_poisson/02_mse2d.5.0']

    """
    if isinstance(fpidx, (list, tuple)):
        fpidx_list = fpidx
    elif isinstance(fpidx, pd.DataFrame):
        fpidx_list = fpidx["fpidx"].tolist()
    elif isinstance(fpidx, pd.Series):
        fpidx_list = fpidx.tolist()
    else:
        assert isinstance(fpidx, str)
        fpidx_list = [fpidx]

    all_fpidxs = []
    for fpidx_ in fpidx_list:
        if fpidx_.count(".") == 0:
            cfg_tree, fidx_, pidx_ = fpidx_, "*", "*"
        elif fpidx_.count(".") == 2:
            cfg_tree, fidx_, pidx_ = fpidx_.split(".")
        else:
            raise ValueError(f"fpidx={fpidx_} is not understood")

        if fidx_ == "*":
            allh5paths = glob.glob(f"{results_dir}/{cfg_tree}_*.h5")
            fidxs = [int(x.split("_")[-1].split(".")[0]) for x in allh5paths]
            fidxs, allh5paths = zip(*sorted(zip(fidxs, allh5paths)))
        else:
            assert fidx_.isdigit()
            fidxs = [int(fidx_)]

        if h5_opener is None:
            all_fpidxs += [f"{cfg_tree}.{fi}.{pidx_}" for fi in fidxs]
        elif pidx_.isdigit():
            all_fpidxs += [f"{cfg_tree}.{fi}.{int(pidx_)}" for fi in fidxs]
        elif pidx_ == "*":
            for fi in fidxs:
                h5fp = h5_opener(cfg_tree, fi)
                pidxs = [
                    int(k.split("P")[1].split("/")[0]) for k in h5fp.keys()
                ]
                all_fpidxs += [f"{cfg_tree}.{fi}.{pi}" for pi in pidxs]
        else:
            raise ValueError(f"Not sure what to do with pidx={pidx_}")

    return all_fpidxs


def get_mtimes(fpidxs):
    fpi2mtime = dict()
    for fpidx in rslv_fpidx(fpidxs):
        cfgtree, fidx, _ = fpidx.split('.')
        fidx = int(fidx)
        hdfpath = f'{results_dir}/{cfgtree}_{fidx:02d}.h5'
        fpi2mtime[fpidx] = getmtime(hdfpath)
    return fpi2mtime

#####################################################
######### The Result Directory Accessories  #########
#####################################################


class resio:
    fp_cahce = odict()
    n_fp = 50
    """
    A helper class for retrieving the results directly
    from the hdf files in the results directory.

    This class does not provide any summarization.

    The methods of this class provide a lot of flexibility, but
    they may be expensive to evaluate. This class is written
    for painless occasional use for either plotting heatmaps,
    inspecting hdf files, etc.

    Global Attributes
    -----------------
    fp_cahce: (OrderedDict) A cache of opened hdf file
        pointers in read mode along with their relavent data
        (e.g., the ioidx and epoch and rng_seed of the stat
        dataframes).

    n_fp: (int) the maximum number of opened hdf file pointer
        to hold in the cache at one time
    """

    def __init__(self, fpidx=None, full=True, resdir=None, driver='core'):
        """
        Class constructor

        Parameters
        ----------
        fpidx: (str, list of str) unresolved `fpidx` values. This will
            be used in the public methods (i.e., `__call__`, `dtypes`,
            `du`, `keys`, `values`, and `items`).

        full: (bool) determines whether the fpidx values should be
            included in the key strings. This will be the treated as
            the default `full` argument for the `keys`, `values`, and
            `items` methods.
        """
        self.fpidx = fpidx
        self.full = full
        self.resdir = resdir if resdir is not None else results_dir
        self.driver = driver

    def get_fpidx_key(
        self, fpidx, key, dtype, rng_seed=None, epoch=None, ret_info=False
    ):
        """
        Gets the key value inside a **single** `fpidx`.

        The only difference between this method and the `__call__` method
        is that it only accepts a single `fpidx` value as a string, whereas
        `__call__` is a wrapper for getting multiple `fpidx` values and
        can accept multiple `fpidx` values provided in a dataframe.

        If the input `epoch` values did not exist in the file, the rows
        with the same rng_seed and closest epoch values will be provided
        instead.

        If the input `rng_seed` values did not exist in the file, a
        python `ValueError` will be raised.

        If `rng_seed` and `epoch` are not None, they must have the same
        length, as they will be zipped together.

        Parameters
        ----------
        fpidx: (str) a single resolved fpidx value
            Example: fpidx = '01_poisson/02_mse2d.0.0'

        key: (str) the hdf key to pull from the h5 file.
            Example: key = '/var/eval/ug/sol/mdl'

        dtype: (str) the data type of the key. It should be one of
            the 'np.arr', 'pd.cat', or 'pd.qnt' values.

        rng_seed: (None or list or np.ndarray) the set of 'rng_seed' values
            to include in the output. If None is provided, all 'rng_seed'
            values will be included in the output.

        epoch: (None, or list or np.ndarray) the set of 'epoch' values to
            include in the output. If None is provided, all 'epoch' values
            will be included in the output.

        ret_info: (bool) whether to return the selection information. If True,
            a data-frame with the 'fpidx', 'epoch', 'rng_seed', and 'ioidx'
            columns will be attached to the output.
        """
        h5fp, h5grp, tdf = self.load_fpidx(fpidx)
        if dtype in ("np.arr",):
            h5ds = h5grp[key]
            msg_ = f'"{key}" does not exist in "{fpidx}"'
            assert key in h5grp, msg_
            msg_ = f'"{key}" is not a leaf dataset in "{fpidx}"'
            assert isinstance(h5ds, h5py.Dataset), msg_
        elif dtype in ("pd.cat", "pd.qnt"):
            cfg_tree, fidx, pidx = fpidx.split(".")
            fidx, pidx = int(fidx), int(pidx)
            hdfpath = f"{results_dir}/{cfg_tree}_{fidx:02d}.h5"
            resdf = pd.read_hdf(hdfpath, key=f"P{pidx:08d}/{key}")
        else:
            raise ValueError(f"dtype={dtype} not defined")

        dsioidx = get_ioidx(h5grp, key)
        tdf = tdf[tdf["ioidx"].isin(dsioidx)]
        if epoch is not None:
            epoch = np.array(epoch).reshape(-1)
            if rng_seed is None:
                unqrs = sorted(tdf["rng_seed"].unique().tolist())
                rng_seed = np.array([rs for ep in epoch for rs in unqrs])
                epoch = np.array([ep for ep in epoch for rs in unqrs])
            else:
                rng_seed = np.array(rng_seed).reshape(-1)
            assert rng_seed.shape == epoch.shape
            utdf = pd.DataFrame(dict(rng_seed=rng_seed, epoch=epoch))
            cat_list = []
            info_list = []
            sorter = np.argsort(dsioidx)
            for rseed, rsutdf in utdf.groupby("rng_seed"):
                adf = tdf[tdf["rng_seed"] == rseed]
                msg_ = f'rng_seed={rseed} does not exist in "{fpidx}"'
                assert adf.shape[0] > 0, msg_
                aa = rsutdf["epoch"].values.reshape(1, -1)
                bb = adf["epoch"].values.reshape(-1, 1)
                ii = np.abs(aa - bb).argmin(axis=0)
                rsinfo = adf.iloc[ii]
                ll = rsinfo["ioidx"].values
                mm = sorter[np.searchsorted(dsioidx, ll, sorter=sorter)]

                if dtype in ("np.arr",):
                    jj, jjxx = np.unique(mm, return_inverse=True)
                    jjas = jj.argsort()
                    jjbs = jjas.argsort()
                    vals_idxd = h5ds[jj[jjas]][jjbs][jjxx]
                elif dtype in ("pd.cat", "pd.qnt"):
                    vals_idxd = resdf.iloc[mm, :]
                else:
                    raise ValueError(f"dtype={dtype} not defined")
                cat_list.append([rsutdf.index.values, vals_idxd])
                info_list.append(rsinfo)
            iii = np.concatenate([x[0] for x in cat_list], axis=0)
            iii_as = iii.argsort()

            info = pd.concat(info_list, axis=0, ignore_index=True)
            info = info.iloc[iii_as].reset_index(drop=True)

            if dtype in ("np.arr",):
                vvv = np.concatenate([x[1] for x in cat_list], axis=0)
                outvals = vvv[iii_as]
            elif dtype in ("pd.cat", "pd.qnt"):
                vvv = pd.concat(
                    [x[1] for x in cat_list], axis=0, ignore_index=True
                )
                outvals = vvv.iloc[iii_as, :]
            else:
                raise ValueError(f"dtype={dtype} not defined")
        else:
            assert rng_seed is None
            info = tdf.reset_index(drop=True)
            if dtype in ("np.arr",):
                outvals = h5ds[:]
            elif dtype in ("pd.cat", "pd.qnt"):
                outvals = resdf
            else:
                raise ValueError(f"dtype={dtype} not defined")

        self.close(self.n_fp)

        out = outvals
        if ret_info:
            out = (out, info)
        return out

    def load_fpidx(self, fpidx, load_pdata=True):
        """
        Creates the hdf file pointer for `fpidx` in read mode, and
        caches it into the `self.fp_cache` variable.

        If `load_pdata` was set as True, the entire 'epoch', 'rng_seed',
        and the 'ioidx' for the 'stat' dataframe will also be loaded
        into the objects's memory.

        Parameters
        ----------
        fpidx: (str) a single fidx-resolved fpidx to load into memory.
            Example: fpidx = '01_poisson/02_mse2d.0.0'

        load_pdata: (bool) whether to load the 'epoch', 'rng_seed', and
            'ioidx' values into memory from the `stat` dataframe of
            the hdf file and partition.
        """
        cfg_tree, fidx, pidx = fpidx.split(".")
        fidx = int(fidx)
        hdfpath = f"{self.resdir}/{cfg_tree}_{fidx:02d}.h5"
        if hdfpath not in self.fp_cahce:
            h5fp = h5py.File(hdfpath, mode="r", driver=self.driver)
            pdata = dict()
            self.fp_cahce[hdfpath] = (h5fp, pdata)
        else:
            h5fp, pdata = self.fp_cahce[hdfpath]

        if pidx == "*":
            h5grp, tdf = None, None
        else:
            pidx = int(pidx)
            if (pidx not in pdata) and load_pdata:
                tdf = pd.read_hdf(hdfpath, key=f"P{pidx:08d}/stat")
                tdf = tdf[["epoch", "rng_seed"]].copy(deep=True)
                stioidx = h5fp[f"P{pidx:08d}/stat/ioidx"][:]
                tdf["fpidx"] = fpidx
                tdf["ioidx"] = stioidx
                pdata[pidx] = tdf
            else:
                tdf = pdata.get(pidx, None)
            h5grp = h5fp[f"P{pidx:08d}"]
        return h5fp, h5grp, tdf

    def close(self, n_fp=0):
        """
        A utility function for closing extra cached hdf file pointers.

        To avoid os complaints, we keep at most `n_fp` hdf files
        open at once.

        All of the open files were opened in the read mode.
        """
        while len(self.fp_cahce) > n_fp:
            for ohdfpath, (oh5fp, opdata) in self.fp_cahce.items():
                break
            oh5fp, opdata = self.fp_cahce.pop(ohdfpath)
            oh5fp.close()
            del opdata

    def rslv_fpidx(self, fpidx):
        """
        a wrapper around the `io_utils.rslv_fpidx` function
        for convinience.
        """

        def h5_opener(cfg_tree, fidx):
            return self.load_fpidx(f"{cfg_tree}.{fidx}.*", load_pdata=False)[0]

        return rslv_fpidx(fpidx, h5_opener)

    def __call__(
        self,
        key,
        dtype=None,
        fpidx=None,
        rng_seed=None,
        epoch=None,
        ret_info=False,
    ):
        """
        A generalization wrapper around 'self.get_fpidx_key'; this method
        can get a dataframe for the `fpidx` input argument, and takes care
         of the grouping, key collection, and output compilation of each
         `fpidx` group.

        See the `get_fpidx_key` for more information.

        The `epoch` and `rng_seed` values can be specified in two ways:
          1. they can be included in the `fpidx` data-frame, or
          2. they can be specified with the `rng_seed` and `epoch` arguments.

        The `epoch` and `rng_seed` values should only be specified in only
        one of the above ways, otherwise the function will complain with a
        `ValueError`.

        If the `epoch` and `rng_seed` values were not specified in any way,
        all 'epoch' and 'rng_seed' values will be returned.

        The `fpidx` argument can be specified in one of three ways:
          1. a single string, or
          2. a list of strings, or
          3. a data-frame with an `fpidx` column, and possibly the
            `epoch` and `rng_seed` values.

        If `fpidx` is provided as a string or a list of strings, they can be
        unresolved (i.e., be incomplete or contain "*" in them).

        Parameters
        ----------
        fpidx: (str or pd.DataFrame) either a single `fpidx` value as string,
            or a df specifying the `fpidx` and possibly the `epoch` and
            `rng_seed` values. The

        key: (str) the hdf key to pull from the h5 file.
            Example: key = '/var/eval/ug/sol/mdl'

        dtype: (str) the data type of the key. It should be one of
            the 'np.arr', 'pd.cat', or 'pd.qnt' values.

        rng_seed: (None or list or np.ndarray) the set of 'rng_seed' values
            to include in the output. If None is provided, all 'rng_seed'
            values will be included in the output.

        epoch: (None, or list or np.ndarray) the set of 'epoch' values to
            include in the output. If None is provided, all 'epoch' values
            will be included in the output.

        ret_info: (bool) whether to return the selection information. If True,
            a data-frame with the 'fpidx', 'epoch', 'rng_seed', and 'ioidx'
            columns will be attached to the output.

        """
        fpidx = fpidx if fpidx is not None else self.fpidx

        if dtype is None:
            dtdict = self.dtypes(full=True, fpidx=fpidx)
            dts = {
                fpikey: dtype_
                for fpikey, dtype_ in dtdict.items()
                if fpikey.split(":")[1] == key
            }
            msg_ = f"not sure what the dtype for {key} is:\n{dts}"
            assert len(set(dts.values())) == 1, msg_
            dtype = list(dts.values())[0]

        if isinstance(fpidx, (str, list)):
            all_fpidxs = self.rslv_fpidx(fpidx)

            all_outs = []
            all_infos = []
            for fpidx_ in all_fpidxs:
                fpout, info = self.get_fpidx_key(
                    fpidx_, key, dtype, rng_seed, epoch, ret_info=True
                )
                all_outs.append(fpout)
                all_infos.append(info)

            info = pd.concat(all_infos, axis=0, ignore_index=True)
            if dtype == "np.arr":
                all_nparr = all(isinstance(x, np.ndarray) for x in all_outs)
                if all_nparr:
                    shapes_set = set(x.shape[1:] for x in all_outs)
                    all_sameshape = len(shapes_set) == 1
                else:
                    all_sameshape = False

                if all_nparr and all_sameshape:
                    sdata = np.concatenate(all_outs, axis=0)
                else:
                    sdata = list(chain.from_iterable(all_outs))
            elif dtype in ("pd.cat", "pd.qnt"):
                sdata = pd.concat(all_outs, axis=0, ignore_index=True)
            else:
                raise ValueError(f"dtype={dtype} not defined")
        else:
            assert isinstance(fpidx, pd.DataFrame)
            df = fpidx
            assert "fpidx" in df.columns
            df = df.reset_index(drop=True).copy(deep=True)
            if rng_seed is not None:
                msg_ = "cannot specify rng_seed when it is in df"
                assert "rng_seed" not in df.columns, msg_
                df["rng_seed"] = rng_seed
            if epoch is not None:
                msg_ = "cannot specify epoch when it is in df"
                assert "epoch" not in df.columns, msg_
                df["epoch"] = epoch
            idx_list = []
            data_list = []
            info_list = []
            for fpidx, fpidf in df.groupby("fpidx"):
                rng_seed_fpi, epoch_fpi = rng_seed, epoch
                if "rng_seed" in fpidf.columns:
                    msg_ = "cannot specify rng_seed when it is in df"
                    assert rng_seed is None, msg_
                    rng_seed_fpi = fpidf["rng_seed"].values
                if "epoch" in fpidf.columns:
                    msg_ = "cannot specify epoch when it is in df"
                    assert epoch is None, msg_
                    epoch_fpi = fpidf["epoch"].values
                vals, info_ = self.get_fpidx_key(
                    fpidx, key, dtype, rng_seed_fpi, epoch_fpi, ret_info=True
                )
                data_list.append(vals)
                info_list.append(info_)

                idxs_ = fpidf.index.values
                n_ireps = vals.shape[0] / len(idxs_)
                assert int(n_ireps) == n_ireps
                idx_list.append(idxs_.repeat(int(n_ireps)))

            idxs = np.concatenate(idx_list, axis=0)
            idxs_as = idxs.argsort(kind="mergesort")
            info = pd.concat(info_list, axis=0, ignore_index=True)
            info = info.iloc[idxs_as].reset_index(drop=True)
            if dtype == "np.arr":
                same_shape = (
                    len(set(np_arr.shape[1:] for np_arr in data_list)) == 1
                )
                if same_shape:
                    data = np.concatenate(data_list, axis=0)
                    sdata = data[idxs_as]
                else:
                    aa = np.cumsum([np_arr.shape[0] for np_arr in data_list])

                    def get_ii(i):
                        i1 = (aa > i).argmax()
                        c = data_list[i1].shape[0]
                        i2 = c - (i - aa[i1])
                        assert i2 >= 0
                        assert i2 < c
                        return i1, i2

                    data = list(chain.from_iterable(data_list))
                    ii_list = [get_ii(i) for i in idxs_as]
                    sdata = [data_list[i1][i2] for i1, i2 in ii_list]
            elif dtype in ("pd.cat", "pd.qnt"):
                data = pd.concat(data_list, axis=0, ignore_index=True)
                sdata = data.iloc[idxs_as]
            else:
                raise ValueError(f"dtype={dtype} not defined")

        out = sdata
        if ret_info:
            out = (out, info)
        return out

    def dtypes(self, full=None, fpidx=None):
        """
        Queies all the keys inside the hdf file and partition
        specified by `fpidx`, and returns a dictionary specifying
        them and their respective dtypes.

        The `fpidx` argument can be unresolved for this method.

        The output will be a dictionary mapping each key to its
        dtype.  The dtype values are one of 'np.arr', 'pd.cat',
        or 'pd.qnt'.

        When `full=True`, the output dictionary keys
        will have f'{fpidx}:{key}' format.

        When `full=False`, the output dictionary keys
        will have a simple f'{key}' format.

        The output will only include keys corresponding to h5py
        datasets or pandas dataframes. In other words, h5py groups
        (i.e., directories) will not be included in the output.

        **Note:** If you suspect that different hdf files/partitions
        have different keys or dtypes, it's best to specify `full=True`
        so that you get a full report of which partition has what
        keys and dtypes.

        Parameters
        ----------
        fpidx: (str, list of str) unresolved `fpidx` values.

        full: (bool) an indicator specifying whether the keys in
          the output must have a leading `fpidx` specifier.

        Outputs
        -------
        info: (OrderedDict) a key-to-dtype mapping of the
          existing keys to their data types.
          Example:

            info = {
              'etc': 'pd.cat',
              'hp': 'pd.cat',
              'stat': 'pd.qnt',
              'mon': 'pd.qnt',
              'var/eval/ug/sol/mdl': 'np.arr',
              'var/eval/ug/sol/gt': 'np.arr',
              'mdl/layer_first.0': 'np.arr',
              ...
            }

            info = {
              '01_poisson/02_mse2d.0.0:etc': 'pd.cat',
              '01_poisson/02_mse2d.0.0:hp': 'pd.cat',
              '01_poisson/02_mse2d.0.0:stat': 'pd.qnt',
              '01_poisson/02_mse2d.0.0:mon': 'pd.qnt',
              '01_poisson/02_mse2d.0.0:mdl/layer_first.0': 'np.arr',
              ...
            }
        """
        fpidx = fpidx if fpidx is not None else self.fpidx
        all_fpidxs = self.rslv_fpidx(fpidx)
        all_fpidxs = list(odict.fromkeys(all_fpidxs))
        full = full if full is not None else self.full

        all_outs = odict()
        for fpidx_ in all_fpidxs:
            h5fp, h5grp, tdf = self.load_fpidx(fpidx_, load_pdata=False)
            key2dtype = get_h5dtypes(h5grp)
            all_outs[f"{fpidx_}:"] = key2dtype

        info = deep2hie(all_outs)
        info = odict([(k.replace(":/", ":", 1), v) for k, v in info.items()])
        if not (full):
            info = odict([(k.split(":")[1], v) for k, v in info.items()])
        return info

    def du(self, verbose=False):
        """
        Returns the disk-usage information.

        The `fpidx` argument can be unresolved for this method.

        The output will be a dictionary mapping each key to its
        number of bytes.

        Parameters
        ----------
        fpidx: (str, list of str) unresolved `fpidx` values.

        verbose: (bool) whether to print the disk-usage information
            in a text table.

        Outputs
        -------
        duinfo: (dict) a dictionary mapping each key to its number
            of bytes.
        """
        fpidx = self.fpidx
        dtypes = self.dtypes(fpidx, full=True)
        duinfo = dict()

        for fpkey, dtype in dtypes.items():
            fpidx, key = fpkey.split(":")
            h5fp, h5grp_, tdf = self.load_fpidx(fpidx, load_pdata=False)
            h5grp = h5grp_[key]
            duinfo[fpkey] = sum(get_h5du(h5grp).values())

        if verbose:
            print_duinfo(duinfo)
        return duinfo

    def keys(self, full=None):
        return list(self.dtypes(full=full).keys())

    def values(self, full=None, **kwargs):
        return list(self.items(full=full, **kwargs).values())

    def items(self, full=None, **kwargs):
        full = full if full is not None else self.full
        dtdict = self.dtypes(full=True)

        keyfps = defaultdict(list)
        for fpikey, dtype in dtdict.items():
            fpidx_, key_ = fpikey.split(":")
            key = fpikey if full else key_
            keyfps[key].append(fpidx_)
        badkeyfps = {k: v for k, v in keyfps.items() if len(v) > 1}
        msg_ = "Multiple files/parts have the same keys. "
        msg_ += 'Specify "full=True" to get all of them.\n'
        for key, fps in badkeyfps.items():
            msg_ += f"{key}: {fps}\n"
        assert len(badkeyfps) == 0, msg_

        out = odict()
        for fpikey, dtype in dtdict.items():
            fpidx_, key_ = fpikey.split(":")
            key = fpikey if full else key_
            out[key] = self.__call__(key_, dtype, fpidx_, **kwargs)
        return out
