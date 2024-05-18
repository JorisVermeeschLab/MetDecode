import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from metdecode.io import load_input_file
from metdecode.model import MetDecode


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, '..', 'preprocessing')


def create_mask(marker_filepath, atlas_filepath, out_filepath, out_filepath2):

    dmrs = []
    with open(marker_filepath, 'r') as f:
        for line in f.readlines():
            elements = line.rstrip().split()
            if len(elements) != 6:
                continue
            cell_type = elements[0].split('.')[1]
            chrom = elements[1]
            start = int(elements[2])
            end = int(elements[3])
            if cell_type == 'Bcell':
                cell_type = 'bcell'
            dmrs.append((cell_type, chrom, start, end))
    cell_types = np.asarray([dmr[0] for dmr in dmrs], dtype=object)


    atlas = dict(np.load(atlas_filepath, allow_pickle=True))
    mask = np.ones(41, dtype=bool)
    mask[-9] = False  # Remove failed sample
    atlas['M'] = atlas['M'][mask]
    atlas['D'] = atlas['D'][mask]
    atlas['cell_type'] = atlas['cell_type'][mask]
    M, D = atlas['M'], atlas['D']
    unique_cell_types = ['BRCA', 'CEAD', 'CESC', 'COAD', 'OV', 'READ', 'bcell', 'cd4tcell', 'cd8tcell', 'erythroblast', 'monocyte', 'naturalkillercell', 'neutrophil']
    cell_type_dict = {cell_type: j for j, cell_type in enumerate(unique_cell_types)}
    cell_type_dict['bcell_SC'] = cell_type_dict['bcell']

    print(M.shape, len(dmrs))
    assert M.shape[1] == len(dmrs)
    assert D.shape[1] == len(dmrs)

    p_values = []
    margins = []
    for k, (cell_type, chrom, start, end) in enumerate(dmrs):
        j = cell_type_dict[cell_type]
        mask = np.ones(M.shape[0], dtype=bool)
        mask[j] = False
        ratios = M[mask, k] / D[mask, k]
        ratio_j = M[j, k] / D[j, k]
        margins.append(np.min(np.abs(ratios - ratio_j)))

        contingency_table = np.asarray([
            [np.sum(M[mask, k]), np.sum(D[mask, k]) - np.sum(M[mask, k])],
            [M[j, k], D[j, k] - M[j, k]]
        ]).T
        p_values.append(scipy.stats.fisher_exact(contingency_table).pvalue)

    margins = np.asarray(margins)
    p_values = np.asarray(p_values)
    p_values[np.isnan(p_values)] = 1
    p_values[np.isinf(p_values)] = 1

    selected = {cell_type: [] for cell_type in set(cell_types)}
    for k in np.argsort(p_values):
        cell_type = cell_types[k]
        if len(selected[cell_type]) < 23:
            selected[cell_type].append(k)
    mask = np.zeros(len(dmrs), dtype=bool)
    for idx in selected.values():
        mask[np.asarray(idx, dtype=int)] = True
    np.save(out_filepath, mask)
    print(np.sum(mask))

    mask = (p_values < 0.001 / len(p_values))
    np.save(out_filepath2, mask)
    print(np.sum(mask))

    return p_values


#p_values = create_mask(
#    os.path.join(ROOT, '..', 'dmrs', 'Annotation_tcga_30_50bp_4CPG.txt'),
#    os.path.join(DATA_DIR, 'atlas30_50bp.npz'),
#    os.path.join(ROOT, '..', 'dmrs', 'mask-balanced-30_50bp.npy'),
#    os.path.join(ROOT, '..', 'dmrs', 'mask-significant-30_50bp.npy')
#)

#p_values = create_mask(
#    os.path.join(ROOT, '..', 'dmrs', 'Annotation_tcga_30_100bp_4CPG.txt'),
#    os.path.join(DATA_DIR, 'atlas30_100bp.npz'),
#    os.path.join(ROOT, '..', 'dmrs', 'mask-balanced-30_100bp.npy'),
#    os.path.join(ROOT, '..', 'dmrs', 'mask-significant-30_100bp.npy')
#)

p_values = create_mask(
    #os.path.join(ROOT, '..', 'dmrs', 'v7_TCGA_regions_4CpG_250l_500bpdist.txt'),
    os.path.join(ROOT, '..', 'dmrs', 'Annotation_Atlas_v7_30_250bp_4CPG.txt'),
    os.path.join(DATA_DIR, 'atlas.npz'),
    os.path.join(ROOT, '..', 'dmrs', 'mask-balanced-none.npy'),
    os.path.join(ROOT, '..', 'dmrs', 'mask-significant-none.npy')
)
