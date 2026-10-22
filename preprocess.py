#!/usr/bin/env python3
"""Convert halo point clouds to padded HDF5 files, split by spatial octant."""
import argparse
from itertools import product
from pathlib import Path

import h5py
import numpy as np

AUX_FIELDS = (
    'is_central_mass',
)


def stellar_stats(attrs):
    for name in ('log_mstar', 'log_sm', 'log_stellar_mass', 'stellar_mass', 'log_ms'):
        if name + '_mean' in attrs and name + '_std' in attrs:
            return float(attrs[name + '_mean']), float(attrs[name + '_std'])
    raise KeyError('Missing stellar-mass normalization attributes: ' + str(list(attrs)))


def split_octants(octants, seed=42):
    """Find disjoint octant groups nearest to 80/10/10 by cloud count."""
    values, counts = np.unique(octants, return_counts=True)
    if len(values) < 3:
        raise ValueError('At least three octants are required')
    perm = np.random.default_rng(seed).permutation(len(values))
    values, counts = values[perm], counts[perm]
    target = np.array([.8, .1, .1]) * len(octants)
    best = min(
        (assignment for assignment in product(range(3), repeat=len(values))
         if len(set(assignment)) == 3),
        key=lambda a: np.square(np.bincount(a, weights=counts, minlength=3) - target).sum(),
    )
    splits = {}
    for split_id, name in enumerate(('train', 'test', 'val')):
        chosen = values[np.asarray(best) == split_id]
        splits[name] = np.flatnonzero(np.isin(octants, chosen))
        print(f'{name}: octants {sorted(chosen.tolist())}, {len(splits[name])} clouds '
              f'({100*len(splits[name])/len(octants):.1f}%)')
    return splits


def export_splits(source, output_dir='/pscratch/sd/v/vmikuni/datasets/halo', seed=42):
    source, output_dir = Path(source), Path(output_dir)
    with h5py.File(source) as f:
        offsets = f['offsets'][:]
        if len(offsets) < 2:
            raise ValueError('Input has no point clouds')
        width = int(np.diff(offsets).max())
        sm_mean, sm_std = stellar_stats(f.attrs)
        mp_mean, mp_std = float(f.attrs['log_mp_mean']), float(f.attrs['log_mp_std'])
        octants = f['octant'][:]
        splits = split_octants(octants, seed)
        gal_index = f['gal_index']

        for split, indices in splits.items():
            folder = output_dir / split
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / (source.stem + '.h5')
            with h5py.File(destination, 'w') as out:
                n = len(indices)
                data = out.create_dataset('data', (n, width, 4), dtype='f4',
                                          chunks=(1, width, 4), compression='lzf')
                target = out.create_dataset('data_pid', (n, width, 1), dtype='f4',
                                            chunks=(1, width, 1), compression='lzf')
                auxiliary = out.create_dataset('global', (n, width, len(AUX_FIELDS)), dtype='f4',
                                               chunks=(1, width, len(AUX_FIELDS)), compression='lzf')
                core = out.create_dataset('is_core', (n, width), dtype='?',
                                          chunks=(1, width), compression='lzf')
                out.create_dataset('pid', data=np.ones(n, dtype=np.int32))
                auxiliary.attrs['feature_names'] = np.asarray(AUX_FIELDS, dtype=h5py.string_dtype())
                out.attrs['source'] = str(source)
                out.attrs['split'] = split
                out.attrs['octants'] = np.unique(octants[indices])
                out.attrs['seed'] = seed
                out.attrs['target_definition'] = 'log10(Mpeak)/10'
                out.attrs['data_features'] = 'x,y,z,log10(stellar_mass)'

                for row, cloud_idx in enumerate(indices):
                    start, end = map(int, offsets[cloud_idx:cloud_idx + 2])
                    count = end - start
                    features = f['features'][start:end]
                    galaxy_rows = gal_index[start:end]
                    x = np.zeros((width, 4), np.float32)
                    y = np.zeros((width, 1), np.float32)
                    a = np.zeros((width, len(AUX_FIELDS)), np.float32)
                    mask = np.zeros(width, bool)
                    x[:count, :3] = features[:, 1:4]
                    x[:count, 3] = features[:, 0] * sm_std + sm_mean
                    y[:count, 0] = (f['target'][start:end] * mp_std + mp_mean) / 10
                    mask[:count] = f['is_core'][start:end]
                    for channel, field in enumerate(AUX_FIELDS):
                        if field in ('host_local', 'group_local'):
                            a[:count, channel] = f[field][start:end]
                        else:
                            # h5py advanced indexing requires sorted indices; indexing
                            # the galaxy array in memory preserves repeated galaxy ids.
                            ids, inverse = np.unique(galaxy_rows, return_inverse=True)
                            a[:count, channel] = f['galaxies'][field][ids][inverse]
                    data[row], target[row], auxiliary[row], core[row] = x, y, a, mask
            print(f'Wrote {destination}')
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output-dir', default='/pscratch/sd/v/vmikuni/datasets/halo')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    export_splits(args.input, args.output_dir, args.seed)
