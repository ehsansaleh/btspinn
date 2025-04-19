import numpy as np
import pandas as pd
import os
import time
import yaml
import glob
import fnmatch
from os.path import exists, getmtime
from itertools import chain
from textwrap import dedent
from collections import defaultdict
from collections import OrderedDict as odict

from bspinn.io_cfg import keyspecs, nullstr
from bspinn.io_cfg import results_dir, summary_dir, source_dir
from bspinn.io_utils import read_pklhdf, get_ioidx
from bspinn.io_utils import get_h5dtypes, pd_concat
from bspinn.io_utils import save_h5data, get_h5du
from bspinn.io_utils import get_mtimes, deep2hie, resio

#####################################################
########## Summarization Utility Functions ##########
#####################################################


def summarize(cfgtree, epstep=2000, keys=None):
    """
    Collects the data-frames from the results directory for a
    particular configuration tree.

    Parameters
    ----------
    cfgtree: (str) The config tree
        Example: cfgtree = '01_poisson/01_btstrp2d'

    epstep: (int) The step size for the epoch averaging.
        For example, when epstep=2000, all rows with
          (a) the same rng_seed, and
          (b) epoch values between 1 and 2000
        will be averaged in the statdf and hpdf.

    keys: (list) a list of strings, each representing a group
        in the hdf partition. It must contain 'hp' and 'stat';
        The 'stat' df will be needed for epoch and rng_seed
        extraction, and 'fpidx' will be inserted into the 'hp' df.
        Example: groups = ['hp', 'stat', 'mdl']

    Output
    ------
    outdict: (dict) a dictionary containing the summarized and
        stacked data from all partitions.
        Example: outdict = {'hp':   pd.DataFrame(...),
                            'stat': pd.DataFrame(...),
                            'mdl':  {'layer1': np.array(...),
                                    'layer2': np.array(...)}}
    """

    groups = ["hp", "stat"] if keys is None else keys
    assert "hp" in groups
    assert "stat" in groups
    grp_specs = []
    for grp_name in groups:

        def matcher(row):
            return fnmatch.fnmatch(grp_name, row[0])

        row = max(keyspecs, key=matcher)
        assert matcher(row), f'Define "{grp_name}" in "grp_specs"'
        grp_specs.append((grp_name, *row[1:]))

    incomp_prtgrpdicts = list()

    #####################################################
    ############## Disk Content Collection ##############
    #####################################################
    print(
        f"Starting Disk Data Collection for {repr(cfgtree)}... ",
        end="",
        flush=True,
    )
    disk_sttime = time.perf_counter()
    # Example: fileidxs = [0, 1, 2, 3]

    rio = resio()
    for fpidx in rio.rslv_fpidx(cfgtree):
        cfgtree_, fidx, pidx = fpidx.split(".")
        fidx, pidx = int(fidx), int(pidx)
        h5fp, h5grp, tdf = rio.load_fpidx(fpidx, load_pdata=True)
        try:
            dtypes = get_h5dtypes(h5grp)
        except RuntimeError as exc:
            msg_ = "Object visitation failed"
            if msg_ in str(exc):
                print(f"{fpidx} will be skipped (corrupted)", flush=True)
                continue
            else:
                raise exc

        grp_dicts = []
        for grp_name, grp_agg in grp_specs:
            ioidx = get_ioidx(h5grp, grp_name)
            grp_types_ = list(
                set(v for k, v in dtypes.items() if k.startswith(grp_name))
            )
            msg_ = f'Many dtypes in "{grp_name}": {grp_types_}'
            assert len(grp_types_) == 1, msg_
            grp_type = grp_types_[0]
            if grp_type in ("pd.cat", "pd.qnt"):
                h5pth = f"{results_dir}/{cfgtree_}_{fidx:02d}.h5"
                # Example: h5pth = '.../results/01_btstrp_00.h5'
                partstrid = f"P{pidx:08d}"
                vals = pd.read_hdf(h5pth, key=f"{partstrid}/{grp_name}")
            elif grp_type in ("pd.pkl",):
                vals = read_pklhdf(h5grp, key=grp_name)
            elif grp_type in ("np.arr",):
                vals = h5grp[grp_name]
            else:
                raise RuntimeError(f"grp_type={grp_type} undef.")

            if grp_name == "hp":
                vals.insert(0, "fpidx", fpidx)

            grp_dict = {
                "name": grp_name,
                "type": grp_type,
                "fpidx": fpidx,
                "agg": grp_agg,
                "ioidx": ioidx,
                "vals": vals,
            }
            # Example:
            #   grp_dict = {'name': 'hp','type': 'pd.cat', 'agg': 'mean',
            #               'fpidx': '01_poisson/01_btstrp2d.0.0',
            #               'vals': pd.DataFrame(...),
            #               'ioidx': np.arange(100)}

            grp_dicts.append(grp_dict)

        for (grp_name, grp_agg), grp_dict in zip(grp_specs, grp_dicts):
            # Adding the 'epoch' and 'rng_seed' columns to
            # each one of the group dictionaries.
            stioidx = tdf["ioidx"].values
            stepochs = tdf["epoch"].values
            strngseeds = tdf["rng_seed"].values

            stio_sorter = np.argsort(stioidx)
            ioidx = grp_dict["ioidx"]
            st_rows = stio_sorter[
                np.searchsorted(stioidx, ioidx, sorter=stio_sorter)
            ]
            epochs = stepochs[st_rows]
            rngseeds = strngseeds[st_rows]
            epmin, epmax = stepochs.min(), stepochs.max()
            unqrs = sorted(set(strngseeds.tolist()))
            grp_dict["epmin"] = epmin  # Example: epmin = 0
            grp_dict["epmax"] = epmax  # Example: epmax = 200_000
            grp_dict["unqrs"] = unqrs  # Example: unqrs = range(100)
            grp_dict["epoch"] = epochs
            grp_dict["rng_seed"] = rngseeds
            # Example:
            #   grp_dict = {'name': 'hp','type': 'pd.cat', 'agg': 'mean',
            #               'vals': pd.DataFrame(...),
            #               'ioidx': np.arange(100),
            #               'epoch': np.arange(100), rng_seed=np.ones(100)}
        incomp_prtgrpdicts.append(grp_dicts)

    # Example:
    #   incomp_prtgrpdicts =
    #                  [[{'name': 'hp',   'type': 'pd.cat', 'agg': 'mean',
    #                     'fpidx': '01_poisson/01_btstrp2d.0.0',
    #                     'vals': pd.DataFrame(...), 'ioidx': np.arange(100)},
    #                    {'name': 'stat', 'type': 'pd.cat', 'agg': 'mean',
    #                     'fpidx': '01_poisson/01_btstrp2d.0.1',
    #                     'vals': pd.DataFrame(...), 'ioidx': np.arange(100)},
    #                    {'name': 'mdl', 'type': 'np.arr', 'agg': 'last',
    #                     'fpidx': '01_poisson/01_btstrp2d.0.2',
    #                     'vals': np.randn(10, 1000), 'ioidx': np.arange(10)}],
    #
    #                   [{'name': 'hp',   'type': 'pd.cat', 'agg': 'mean',
    #                     'fpidx': '01_poisson/01_btstrp2d.4.0',
    #                     'vals': pd.DataFrame(...), 'ioidx': np.arange(100)},
    #                    {'name': 'stat', 'type': 'pd.cat', 'agg': 'mean',
    #                     'fpidx': '01_poisson/01_btstrp2d.4.1',
    #                     'vals': pd.DataFrame(...), 'ioidx': np.arange(100)},
    #                    {'name': 'mdl', 'type': 'np.arr', 'agg': 'last',
    #                     'fpidx': '01_poisson/01_btstrp2d.4.2',
    #                     'vals': np.randn(10, 1000), 'ioidx': np.arange(10)}],
    #                   ...]

    #####################################################
    ############## Merging Incomplete Runs ##############
    #####################################################
    g2dict = odict()
    for grp_dicts in incomp_prtgrpdicts:
        for grp_dict in grp_dicts:
            cfgtree_, fidx_, pidx_ = grp_dict["fpidx"].split(".")
            grpid = (grp_dict["name"], cfgtree_, int(fidx_))
            g2dict.setdefault(grpid, [])
            g2dict[grpid].append(grp_dict)

    # Example:
    #   g2dict = {('hp', 'pd.cat', 'last', '01_poisson/01_btstrp2d', 0):
    #               [{'name': 'hp', 'type': 'pd.cat', 'agg': 'last',
    #                 'fpidx': '01_poisson/01_btstrp2d.0.0',
    #                'vals': pd.DataFrame(...),
    #                'ioidx': np.arange(0,     1000)},
    #                {'name': 'hp', 'type': 'pd.cat', 'agg': 'last',
    #                 'fpidx': '01_poisson/01_btstrp2d.0.1',
    #                 'vals': pd.DataFrame(...),
    #                 'ioidx': np.arange(1000, 2000)},
    #             ....
    #             }

    # Finding the minimum and maximum ioidx value in each fpidx
    fp2ioimins = defaultdict(list)
    fp2ioimaxs = defaultdict(list)
    for grp_dicts in incomp_prtgrpdicts:
        for grp_dict in grp_dicts:
            fp2ioimins[grp_dict["fpidx"]].append(grp_dict["ioidx"].min())
            fp2ioimaxs[grp_dict["fpidx"]].append(grp_dict["ioidx"].max())

    fp2ioimm = {
        fpi: (min(fp2ioimins[fpi]), max(fp2ioimaxs[fpi])) for fpi in fp2ioimins
    }
    # Example:
    # fp2ioimm = {'01_poisson/01_btstrp2d.0.0': (0,      100000),
    #             '01_poisson/01_btstrp2d.0.1': (100001, 200000),
    #             '01_poisson/01_btstrp2d.0.2': (0,      120000),
    #             '01_poisson/01_btstrp2d.0.3': (120001, 200000),
    #             ...}

    fp2ioimm2 = defaultdict(list)
    for fpidx, (ioidx_min, ioidx_max) in fp2ioimm.items():
        cfg_tree_, fidx, pidx = fpidx.split(".")
        fidx, pidx = int(fidx), int(pidx)
        fp2ioimm2[f"{cfg_tree_}.{fidx}"].append((pidx, ioidx_min, ioidx_max))
    fp2ioimm2 = {k: sorted(v) for k, v in fp2ioimm2.items()}
    # Example:
    #   fp2ioimm2 = {'01_poisson/01_btstrp2d.0': [(0,        0, 100000),
    #                                             (1,   100001, 200000),
    #                                             (2,   200001, 250000)],
    #                '01_poisson/01_btstrp2d.1': [(0,        0, 110000),
    #                                             (1,   110001, 220000),
    #                                             (2,   220001, 250000)],
    #                ...}

    fpidxgrps = []
    for fidx_, ioidxinfo in fp2ioimm2.items():
        lastmax = None
        for pidx, ioidx_min, ioidx_max in ioidxinfo:
            if (lastmax is None) or (ioidx_min <= lastmax):
                fpidxgrps.append([])
                lastmax = ioidx_max
            fpidx_ = f"{fidx_}.{pidx}"
            fpidxgrps[-1].append(fpidx_)
    # fpidxgrps = [['01_poisson/01_btstrp2d.0.0',
    #               '01_poisson/01_btstrp2d.0.1',
    #               '01_poisson/01_btstrp2d.0.2'],
    #              ['01_poisson/01_btstrp2d.1.0',
    #               '01_poisson/01_btstrp2d.1.1',
    #               '01_poisson/01_btstrp2d.1.2'],
    #              ...]

    pgrpdicts = []
    for grpid, gdict_list in g2dict.items():

        def sortkey(gd):
            return int(gd["fpidx"].split(".")[-1])

        gdict_list = sorted(gdict_list, key=sortkey)
        for fpidxgrp in fpidxgrps:
            gdict_grp = [
                grp_dict
                for grp_dict in gdict_list
                if grp_dict["fpidx"] in fpidxgrp
            ]
            if len(gdict_grp) > 0:
                pgrpdicts.append(gdict_grp)
    # Example:
    #      pgrpdicts = [[{'name': 'hp',   'type': 'pd.cat', 'agg': 'last',
    #                     'fpidx': ['01_poisson/01_btstrp2d.0.0'],
    #                     'vals': pd.DataFrame(...),
    #                     'ioidx': np.arange(0, 100)},
    #                    {'name': 'hp',   'type': 'pd.cat', 'agg': 'last',
    #                     'fpidx': ['01_poisson/01_btstrp2d.0.1'],
    #                     'vals': pd.DataFrame(...),
    #                     'ioidx': np.arange(100, 200)}],
    #                   [{'name': 'stat', 'type': 'pd.cat', 'agg': 'mean',
    #                     'fpidx': ['01_poisson/01_btstrp2d.0.0'],
    #                     'vals': pd.DataFrame(...),
    #                     'ioidx': np.arange(0, 100)},
    #                    {'name': 'stat', 'type': 'pd.cat', 'agg': 'mean',
    #                     'fpidx': ['01_poisson/01_btstrp2d.0.0'],
    #                     'vals': pd.DataFrame(...),
    #                     'ioidx': np.arange(0, 100)}],
    #                   ...]

    partgrpdicts = []
    for gdl_ in pgrpdicts:
        gdl = [gd.copy() for gd in gdl_]
        grp_names = [gd.pop("name") for gd in gdl]
        grp_types = [gd.pop("type") for gd in gdl]
        grp_aggs = [gd.pop("agg") for gd in gdl]
        grp_fpidxs = [gd.pop("fpidx") for gd in gdl]
        grp_ioidxs = [gd.pop("ioidx") for gd in gdl]
        grp_valss = [gd.pop("vals") for gd in gdl]
        grp_epmins = [gd.pop("epmin") for gd in gdl]
        grp_epmaxs = [gd.pop("epmax") for gd in gdl]
        grp_unqrss = [gd.pop("unqrs") for gd in gdl]
        grp_epochs = [gd.pop("epoch") for gd in gdl]
        grp_rng_seeds = [gd.pop("rng_seed") for gd in gdl]

        assert len(set(grp_names)) == 1
        assert len(set(grp_types)) == 1
        assert len(set(grp_aggs)) == 1
        assert len(set([tuple(x.split(".")[:2]) for x in grp_fpidxs])) == 1
        assert all(len(gd) == 0 for gd in gdl), gdl

        grp_type = grp_types[0]
        grp_name = grp_names[0]
        grp_agg = grp_aggs[0]
        grp_epmin = min(grp_epmins)
        grp_epmax = max(grp_epmaxs)
        grp_unqrs = sorted(set(chain.from_iterable(grp_unqrss)))
        grp_epoch = np.concatenate(grp_epochs, axis=0)
        grp_rng_seed = np.concatenate(grp_rng_seeds, axis=0)

        grp_ioidx = np.concatenate(grp_ioidxs, axis=0)
        if grp_type in ("pd.cat", "pd.qnt", "pd.pkl"):
            valscat = pd_concat(grp_valss, axis=0, ignore_index=True)
        elif grp_type in ("np.arr",):
            # Example:
            #     grp_name = 'var/eval/ug'
            #     h5grp = h5fp['P00000000/var/eval/ug']
            #     npkeys = ['sol/gt', 'sol/mdl', 'sol/trg']
            dsetdtypes_ = [
                list(get_h5dtypes(h5grp).items()) for h5grp in grp_valss
            ]
            key2dtype_ = dict(list(chain.from_iterable(dsetdtypes_)))

            npkey_arrs = defaultdict(list)
            for npkey, dtype in key2dtype_.items():
                assert dtype == "np.arr"
                for h5grp in grp_valss:
                    h5arr = h5grp.get(npkey, None)
                    if h5arr is not None:
                        npkey_arrs[npkey].append(h5arr[:])
            valscat = {
                npkey: np.concatenate(arrlist, axis=0)
                for npkey, arrlist in npkey_arrs.items()
            }
        else:
            raise ValueError(f"grp_type={grp_type} not defined")

        if grp_name == "hp":
            aaa = [
                tuple(max(fpidxgrps, key=lambda fpis: int(fpidx in fpis)))
                for fpidx in grp_fpidxs
            ]
            assert len(set(aaa)) == 1
            fpidxg = aaa[0][0]
            valscat.insert(0, "fpidxgrp", fpidxg)

        grp_dict = {
            "name": grp_name,
            "type": grp_type,
            "agg": grp_agg,
            "epmin": grp_epmin,
            "epmax": grp_epmax,
            "unqrs": grp_unqrs,
            "epoch": grp_epoch,
            "rng_seed": grp_rng_seed,
            "fpidx": grp_fpidxs,
            "vals": valscat,
            "ioidx": grp_ioidx,
        }

        partgrpdicts.append(grp_dict)

    # Example:
    #    partgrpdicts = [{'name': 'hp',   'type': 'pd.cat', 'agg': 'last',
    #                     'fpidx': ['01_poisson/01_btstrp2d.0.0',
    #                               '01_poisson/01_btstrp2d.0.1'],
    #                     'vals': pd.DataFrame(...), 'ioidx': np.arange(100)},
    #                    {'name': 'stat', 'type': 'pd.cat', 'agg': 'mean',
    #                     'fpidx': ['01_poisson/01_btstrp2d.0.0',
    #                               '01_poisson/01_btstrp2d.0.1'],
    #                     'vals': pd.DataFrame(...), 'ioidx': np.arange(100)},
    #                    {'name': 'mdl', 'type': 'np.arr', 'agg': 'last',
    #                     'fpidx': ['01_poisson/01_btstrp2d.0.0',
    #                               '01_poisson/01_btstrp2d.0.1'],
    #                     'vals': np.randn(20, 1000), 'ioidx': np.arange(20)}],
    #                   ...]

    print(f"Finished in {time.perf_counter()-disk_sttime:.2f} sec.",
          flush=True)

    #####################################################
    ############### Summarizing the Values ##############
    #####################################################
    print(f"Starting to summarize the data for {cfgtree}... ",
          end='', flush=True)
    smry_sttime = time.perf_counter()
    # Adding the summarized values versions to each grp_dict
    for grp_dict in partgrpdicts:
        epmin, epmax = grp_dict["epmin"], grp_dict["epmax"]
        # Example: epmin, epmax = 0, 200_000

        seeds_unq = grp_dict["unqrs"]
        # Example:
        #    seeds_unq = [0, 1000, ..., 99000]

        epochs = grp_dict["epoch"]
        # Example:
        #    epochs = np.array([0, 0,...,200_000, 200_000])
        #    epochs.shape == (8100,)

        rng_seeds = grp_dict["rng_seed"]
        # Example:
        #    rng_seeds = np.array([0, 1000, ...., 98000, 99000])
        #    rng_seeds.shape == (8100,)

        grp_type = grp_dict["type"]
        # Example:
        #    grp_type = 'pd.cat'
        #    grp_type = 'pd.qnt'
        #    grp_type = 'np.arr'

        grp_agg = grp_dict["agg"]
        # Example:
        #    grp_agg = 'last'
        #    grp_agg = 'mean'

        n_epbins = int(np.ceil(epmax / epstep))
        n_epbins = n_epbins - int(np.ceil(epmin / epstep)) + 1
        # `n_epbins` is the number of epoch bins. Assuming `epstep = 2000`, We
        # coniser all epoch values in (1, 2, ..., 2000) as 2000 for
        # summarization.
        # Example: n_epbins = 200

        n_seeds_unq = len(seeds_unq)
        # n_seeds_unq is the number of unique rng seeds.

        # The following tries to find the row indexes for each (epoch, seed)
        # bin. The result will be stored in `bin_idxs`, which is a list with
        # `n_epbins * n_seeds_unq` inner lists, each containing some row
        # indexes.
        epbins = np.ceil(epochs / epstep).astype(int)
        rsbins = np.searchsorted(seeds_unq, rng_seeds, side="left")
        bin_idxs = [[] for _ in range(n_epbins * n_seeds_unq)]
        for idx, (epbin, rsbin) in enumerate(zip(epbins, rsbins)):
            bin_idxs[rsbin * n_epbins + epbin].append(idx)
        # Example:
        #   bin_idxs = [[0],
        #               [100, 200, 300, ...],
        #               ...,
        #               [..., 199899, 199999, 200099]]

        # Some bins may be empty, so they may have to borrow their values from
        # the previous epoch.
        assert all(
            len(idxs) > 0
            for bin_, idxs in enumerate(bin_idxs)
            if bin_ % n_epbins == 0
        ), "epoch = 0 must be non-empty"
        for bin_, idxs in enumerate(bin_idxs):
            if len(idxs) == 0:
                bin_idxs[bin_] = bin_idxs[bin_ - 1]

        # Giving each (epoch, seed) pair a bin number, and storing
        # it similar to the bin_idxs structure
        bins = [[bin_] * len(idxs) for bin_, idxs in enumerate(bin_idxs)]
        # Example:
        #    bins = [[0],
        #            [1, 1, 1, 1, ...],
        #            ...,
        #            [..., 10099, 10099, 10099]]

        # Chaining the `bin_idxs` and `bins_ch` to be a simple list of ints.
        bin_idxs_ch = list(chain.from_iterable(bin_idxs))
        bins_ch = list(chain.from_iterable(bins))
        # Example:
        #      bin_idxs_ch = [0, 100, 200, 300, ..., 199899, 199999, 200099]
        #      bins =        [0,   1,   1,   1, ...,  10099,  10099,  10099]

        if grp_type in ("pd.cat",):
            # Unfortunately, there is a problem with pandas being super
            # slow when grouping and summarizing categorical data-frames.
            # Since the hyper-parameter and etc data-frames are usually the
            # ones using 'pd.cat' types, I'm assuming they have identical
            # rows. If this did not hold at any point, just disable this case
            # and let the next case take care of the summarization (which will
            # make things slower, but it will at least work safely).
            df = grp_dict["vals"]
            row1 = df.iloc[0, :]
            assert (df.eq(row1, axis=1) | (df.isna() & row1.isna())).values.all()
            sudf = df.iloc[: n_epbins * n_seeds_unq, :]
            grp_dict["smry"] = sudf
        elif grp_type in ("pd.qnt", "pd.cat"):
            # Copying and re-arranging the rows such that the same bin
            # rows would be placed next to each other.
            df = grp_dict["vals"].iloc[bin_idxs_ch].copy(deep=False)

            # Now that the bin members are correctly ordered, we will insert
            # the 'bins' column.
            df["bins"] = bins_ch

            # Summarizing each bin group using either `grp_agg='last'` or
            # `grp_agg='mean'` aggregation rules.
            sudf = df.groupby("bins").agg(grp_agg).reset_index(drop=True)

            # The 'mean' aggregation turns epochs into float numbers, and
            # the mean epoch may fall anywhere. To address this, we will
            # manually re-calculate the epochs, and set them to the last
            # epoch value in the bin.
            if "epoch" in sudf.columns:
                a = np.arange(n_epbins * n_seeds_unq)
                agg_epoch = epstep * (a % n_epbins).astype(int)
                sudf["epoch"] = agg_epoch

            # Applying the same for 'rng_seed' even though it should not
            # be necessary; All rows in the same bin must have the same
            # `rng_seed`, so even mean shouldn't change them. We're just
            # performing a sanity check assertion here, and dtype casting.
            if "rng_seed" in sudf.columns:
                a = np.arange(n_epbins * n_seeds_unq)
                agg_rseeds = np.array(seeds_unq)[(a // n_epbins).astype(int)]
                assert (agg_rseeds == sudf["rng_seed"]).all()
                sudf["rng_seed"] = agg_rseeds
            grp_dict["smry"] = sudf
        elif grp_type in ("np.arr",):
            # Applying the summarization on the numpy arrays
            smry_dict = dict()
            for key, arr in grp_dict["vals"].items():
                if grp_agg == "mean":
                    rows = [
                        np.stack(arr[idxs], axis=0).mean(axis=0)
                        for bin_, idxs in enumerate(bin_idxs)
                    ]
                    suarr = np.concatenate(rows, axis=0)
                elif grp_agg == "last":
                    lidxs = [idxs[-1] for idxs in bin_idxs]
                    suarr = arr[lidxs]
                else:
                    raise ValueError(f"grp_agg={grp_agg} not defined")
                smry_dict[key] = suarr

            # Example:
            #    smry_dict = {'sol/gt':  np.array(...),
            #                 'sol/mdl': np.array(...),
            #                 'sol/trg': np.array(...),}
            grp_dict["smry"] = smry_dict
        else:
            raise ValueError(f"grp_type={grp_type} not defined")

    print(
        f"Finished in {time.perf_counter()-smry_sttime:.2f} sec.", flush=True
    )

    #####################################################
    ####### Concatenating the Summarized DF/Arrays ######
    #####################################################
    print(f"Stacking the data for {cfgtree}... ", end="", flush=True)
    stck_sttime = time.perf_counter()
    # Here, we concatenate all of the summarized df/arrays from
    # different partitions into a single df/arrays for each name.

    # The only inputs to this part are `partgrpdicts` and `grp_specs`.
    # `grp_specs` is not so essential here, so you can probably make
    # due with only the `partgrpdicts` variable.
    concatdict = dict()
    for grp_i, (grp_name, grp_agg) in enumerate(grp_specs):
        # Example:
        #   grp_name = 'hp'
        #   grp_type = 'pd.cat'
        #   grp_agg =  'last'
        grp_dicts = [
            grp_dict
            for grp_dict in partgrpdicts
            if grp_dict["name"] == grp_name
        ]
        grp_types = list(set(grp_dict["type"] for grp_dict in grp_dicts))
        assert len(grp_types) == 1
        assert all(grp_dict["agg"] == grp_agg for grp_dict in grp_dicts)
        grp_type = grp_types[0]

        smry_list = [grp_dict["smry"] for grp_dict in grp_dicts]
        # Example:
        #   smry_list = [df1, df2, ..., df38]
        #   smry_list = [{'sol/gt':  a1, 'sol/mdl':  b1},
        #                {'sol/gt':  a2, 'sol/mdl':  b2},
        #                ...,
        #                {'sol/gt': a38, 'sol/mdl': b38}]

        fpidxs_list = [grp_dict["fpidx"] for grp_dict in grp_dicts]
        # Example:
        #   fpidxs_list = [['01_poisson/01_btstrp2d.0.1'],
        #                  ...,
        #                  ['01_poisson/01_btstrp2d.7.3']]
        if grp_type in ("pd.cat", "pd.qnt", "pd.pkl"):
            df = pd.concat(smry_list, axis=0, ignore_index=True)
            if grp_type == "pd.cat":
                for col in df.columns:
                    df[col] = df[col].astype("category")
                    df[col] = df[col].cat.add_categories(nullstr)
                    df[col] = df[col].fillna(nullstr)
            smry_cat = df
        elif grp_type in ("np.arr",):
            all_leafs = set(
                chain.from_iterable(np_dict.keys() for np_dict in smry_list)
            )
            # Example: all_leafs = {'sol/gt', 'sol/mdl', 'sol/trg'}
            smry_cat = dict()
            for leaf in all_leafs:
                np_arrs = [np_dict.get(leaf, None) for np_dict in smry_list]
                empty_fpidxs = list(
                    chain.from_iterable(
                        fpidxs
                        for fpidxs, np_arr in zip(fpidxs_list, np_arrs)
                        if np_arr is None
                    )
                )
                msg1_ = f"""
                Not all partitions have the {leaf} leaf in "{grp_name}".
                Partitions "{empty_fpidxs}" lack "{leaf}" in them.
                In order to concatenate all {leaf} arrays, they must
                be present in all partitions."""

                shapes_ = {
                    tuple(fpidxs): np_arr.shape[1:]
                    for fpidxs, np_arr in zip(fpidxs_list, np_arrs)
                    if np_arr is not None
                }
                can_concat = len(set(shapes_.values())) == 1
                msg2_ = f"""
                Warning: Not all partitions have the same {leaf} shape in
                         "{grp_name}". In order to concatenate them, {leaf}
                         must have the same shape in all partitions. I will
                         just chain them as lists.
                         shapes = {shapes_}"""

                assert all(x is not None for x in np_arrs), dedent(msg1_)
                # assert can_concat, dedent(msg2_)

                if can_concat:
                    smry_cat[leaf] = np.concatenate(np_arrs, axis=0)
                else:
                    print(dedent(msg2_))
                    smry_cat[leaf] = list(
                        chain.from_iterable(list(np_arr) for np_arr in np_arrs)
                    )
        else:
            raise RuntimeError(f"grp_type={grp_type} undef.")

        concatdict[grp_name] = smry_cat

    # Example:
    #    concatdict = {'hp':   pd.DataFrame(...),
    #                  'stat': pd.DataFrame(...),
    #                  'mdl':  {'layer1': np.array(...),
    #                           'layer2': np.array(...)}
    #                  }
    print(f"Finished in {time.perf_counter()-stck_sttime:.2f} sec.",
          flush=True)

    # To absolve all np.arr dictionaries into the highest level, we apply a
    # deep2hie to the final dictionary.
    outdict = deep2hie(concatdict)

    return outdict


if __name__ == "__main__":
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('--lazy', action='store_true')
        my_parser.add_argument('-e', '--exps',   action='store', type=str, default='*', required=False)
        args = my_parser.parse_args()
        be_lazy = args.lazy
        exp_pats = args.exps.split(',')
    else:
        be_lazy = True
        exp_pats = ['*']

    # Compiling the `z*_expspec.yml` into a single data-frame
    expdflst = []
    for expspecpath in glob.glob(f'{source_dir}/z*_expspec.yml'):
        with open(expspecpath, 'r') as fp:
            expspec = yaml.safe_load(fp)
        for expname, rowslist in expspec.items():
            expdflst += [{'experiment': expname, **rowdict} for rowdict in rowslist]
    expdf1 = pd.DataFrame(expdflst)
    assert len(expdflst) > 0, f'No experiment spec files at "{source_dir}/z*_expspec.yml"'

    keep_exps = [experiment for experiment in expdf1['experiment'].unique().tolist()
        if any(fnmatch.fnmatch(experiment, pat) for pat in exp_pats)]
    expdf2 = expdf1[expdf1['experiment'].isin(keep_exps)]
    print(f'The experiments to summarize are {keep_exps}\n', flush=True)

    # Generating the summary files for each experiment
    for experiment, edf in expdf2.groupby('experiment'):
        smrypath = f'{summary_dir}/{experiment}.h5'

        do_summarize = True
        if be_lazy and exists(smrypath):
            resmtimes = get_mtimes(edf['fpidx'].tolist())
            smrymtime = getmtime(smrypath)
            do_summarize = max(resmtimes.values()) > smrymtime

        if not do_summarize:
            continue

        print(f'--> Starting experiment "{experiment}"\n')
        savedata = dict()
        for key, fpidf in edf.groupby('smrykey'):
            fpidxs = fpidf['fpidx'].tolist()
            if key == '':
                savedata.update(summarize(fpidxs))
            else:
                savedata[key] = summarize(fpidxs)
            print('')

        print(f'Writing "{smrypath}".\n', flush=True)
        os.makedirs(summary_dir, exist_ok=True)
        if exists(smrypath):
            os.remove(smrypath)
        save_h5data(deep2hie(savedata), smrypath)
        get_h5du(smrypath, verbose=True, detailed=False)
        print('*' * 100)
