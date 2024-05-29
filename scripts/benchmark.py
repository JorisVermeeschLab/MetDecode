import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from typing import List, Tuple
import argparse
import shutil
import sys
sys.path.insert(0, '..')

import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt

from metdecode.model import MetDecode
from metdecode.benchmarking import run_cancerlocator
from metdecode.io import load_input_file


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(ROOT, 'results')


def process(method, experiment, markers, marker_filter, atlas_with_rep=False, n_unknowns=0, override_=False):
    N_UNKNOWNS = n_unknowns
    folder = os.path.join(DATA_DIR, markers, marker_filter)

    # Create output dir
    if method == 'metdecode':
        method_name = f'metdecode-{N_UNKNOWNS}_unk'
    elif method == 'celfie':
        method_name = f'celfie-{N_UNKNOWNS}_unk'
    else:
        method_name = method
    out_folder = os.path.join(OUT_DIR, experiment, markers, marker_filter, 'normal-atlas' if (not atlas_with_rep) else 'atlas-with-rep')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    out_filepath = os.path.join(out_folder, f'{method_name}.alpha.csv')

    if not override_:
        if os.path.exists(out_filepath):
            print(f'Already processed: {out_filepath}')
            return

    INPUT_ATLAS = os.path.join(folder, 'atlas-with-rep.tsv' if atlas_with_rep else 'atlas.tsv')
    if experiment == 'too-cbc':
        INPUT_TEST_DATA = os.path.join(folder, 'gdna-for-cbc-and-too.tsv')
    elif experiment == 'cfdna':
        INPUT_TEST_DATA = os.path.join(folder, 'cfdna.tsv')
    elif experiment == 'all':
        INPUT_TEST_DATA = os.path.join(folder, 'all.tsv')
    elif experiment == 'all-insilico':
        INPUT_TEST_DATA = os.path.join(folder, 'all-insilico.tsv')
    elif experiment in {'Br62', 'Br66', 'Cer77', 'Cer81', 'Colo', 'Colo9345', 'Ov79', 'Ov9433'}:
        INPUT_TEST_DATA = os.path.join(folder, f'insilico_{experiment}.tsv')
    else:
        raise NotImplementedError(f'Unknown experiment "{experiment}"')

    # Load input data
    M_atlas, D_atlas, cell_types, marker_names = load_input_file(INPUT_ATLAS)
    M_cfdna, D_cfdna, sample_names, marker_names2 = load_input_file(INPUT_TEST_DATA)
    for marker1, marker2 in zip(marker_names, marker_names2):
        marker1 = marker1.split('-')
        marker2 = marker2.split('-')
        if (marker1[0] != marker2[0]) or (marker1[1] != marker2[1]):
            raise ValueError(f'Marker regions differ in the two input files: {marker1} and {marker2}')
    y_atlas = np.zeros(len(cell_types), dtype=int)
    for j in range(len(cell_types)):
        if 'BRCA' in cell_types[j]:
            y_atlas[j] = 1
        elif 'OV' in cell_types[j]:
            y_atlas[j] = 2
        elif 'CESC' in cell_types[j]:
            y_atlas[j] = 3
        elif 'CEAD' in cell_types[j]:
            y_atlas[j] = 3
        elif 'COAD' in cell_types[j]:
            y_atlas[j] = 4
        elif 'READ' in cell_types[j]:
            y_atlas[j] = 4
        else:
            y_atlas[j] = 0
    y_cfdna = np.zeros(len(sample_names), dtype=int)
    for i in range(len(sample_names)):
        if 'Breast' in sample_names[i]:
            y_cfdna[i] = 1
        elif 'Ovarian' in sample_names[i]:
            y_cfdna[i] = 2
        elif 'Cervical' in sample_names[i]:
            y_cfdna[i] = 3
        elif 'Colorectal' in sample_names[i]:
            y_cfdna[i] = 4
        else:
            y_cfdna[i] = 0

    alpha = None
    if method == 'cancerlocator':
        M_atlas, D_atlas = MetDecode.add_pseudo_counts(M_atlas, D_atlas)
        tmp_out_filepath = run_cancerlocator(
            'CancerLocator.jar', 'tmp', cell_types, M_atlas / D_atlas, sample_names, M_cfdna, D_cfdna,
            n_threads=32
        )
        shutil.copyfile(tmp_out_filepath, os.path.join(out_folder, f'{method}.output.tsv'))
    elif method == 'metdecode':
        method = f'metdecode-{N_UNKNOWNS}_unk'
        out_filepath = os.path.join(out_folder, f'{method}.alpha.csv')
        model = MetDecode(M_atlas, D_atlas, M_cfdna, D_cfdna, n_unknown_tissues=N_UNKNOWNS)
        alpha = model.deconvolute()
        out_cell_types = list(cell_types)
        for j in range(N_UNKNOWNS):
            out_cell_types.append(f'Unknown-{j + 1}')
    elif method == 'nnls':
        M_atlas, D_atlas = MetDecode.add_pseudo_counts(M_atlas, D_atlas)
        M_cfdna, D_cfdna = MetDecode.add_pseudo_counts(M_cfdna, D_cfdna)
        R_atlas = M_atlas / D_atlas
        R_cfdna = M_cfdna / D_cfdna
        alpha = []
        for i in range(M_cfdna.shape[0]):
            x, residuals = scipy.optimize.nnls(R_atlas.T, R_cfdna[i, :], maxiter=3000)
            alpha.append(x / np.sum(x))
        alpha = np.asarray(alpha)
        out_cell_types = list(cell_types)
    elif method == 'qp':
        M_atlas, D_atlas = MetDecode.add_pseudo_counts(M_atlas, D_atlas)
        M_cfdna, D_cfdna = MetDecode.add_pseudo_counts(M_cfdna, D_cfdna)
        R_atlas = M_atlas / D_atlas
        R_cfdna = M_cfdna / D_cfdna
        A = R_atlas
        B = R_cfdna
        reference_matrix = A.T
        cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})
        alpha = np.zeros((R_cfdna.shape[0], R_atlas.shape[0]))
        x0 = np.zeros(R_atlas.shape[0])
        for i in range(R_cfdna.shape[0]):
            def loss(x):
                diff = np.dot(R_atlas.T, x)
                return np.sum(np.square(diff - R_cfdna[i, :]))

            res = scipy.optimize.minimize(
                    loss, x0, method='SLSQP', constraints=cons,
                    bounds=[(0, 1) for j in range(R_atlas.shape[0])],
                    options={'disp': False, 'maxiter': 3000})
            alpha[i, :] = res.x
        out_cell_types = list(cell_types)
    elif method == 'celfie':
        method = f'celfie-{N_UNKNOWNS}_unk'
        from metdecode.celfie import celfie_deconvolute
        alpha, _ = celfie_deconvolute(M_cfdna, D_cfdna, M_atlas, D_atlas, max_n_iter=1000, n_runs=1, convergence_rate=0.0001, n_unknown_tissues=N_UNKNOWNS, verbose=True)
        out_cell_types = list(cell_types)
        for j in range(N_UNKNOWNS):
            out_cell_types.append(f'Unknown-{j + 1}')
    else:
        raise NotImplementedError(f'Unknown deconvolution algorithm "{method}"')

    # Results are stored in alpha, where alpha[i, j] is the contribution of tissue j to methylation profile i
    if alpha is not None:
        with open(out_filepath, 'w') as f:
            f.write(','.join(['Sample'] + list(out_cell_types)) + '\n')
            for i, sample_name in enumerate(sample_names):
                f.write(sample_name)
                for value in alpha[i, :]:
                    percentage = 100. * value
                    f.write(f',{percentage:.3f}')
                f.write('\n')
        print(f'Deconvolution results stored at {out_filepath}.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=['metdecode', 'nnls', 'qp', 'celfie', 'qp', 'cancerlocator'], help='Deconvolution algorithm')
    parser.add_argument('experiment', type=str, choices=['too-cbc', 'cfdna', 'Br62', 'Br66', 'Cer77', 'Cer81', 'Colo', 'Colo9345', 'Ov79', 'Ov9433', 'all-insilico', 'all'], help='Input samples to be used')
    parser.add_argument('markers', type=str, choices=['30_250bp', '30_50bp', '30_100bp'])
    parser.add_argument('marker_filter', type=str, choices=['all-markers', 'balanced', 'significant'])
    parser.add_argument('--atlas-with-rep', action='store_true', help='Whether not to aggregate the replicates in the atlas')
    parser.add_argument('--n-unknowns', type=int, default=0, help='Number of unknown cell types to infer with MetDecode or CelFIE')
    args = parser.parse_args()

    process(
        args.method,
        args.experiment,
        args.markers,
        args.marker_filter,
        atlas_with_rep=args.atlas_with_rep,
        n_unknowns=args.n_unknowns
    )


if __name__ == '__main__':

    # main()

    #for method in ['cancerlocator']:
    #    for experiment in ['all']:
    #        for markers in ['30_250bp']:
    #            for marker_filter in ['all-markers', 'balanced', 'significant']:
    #                for atlas_with_rep in [True]:
    #                    process(method, experiment, markers, marker_filter, atlas_with_rep=atlas_with_rep, n_unknowns=0)

    for method in ['celfie']:
        for experiment in ['cfdna']:
            for markers in ['30_250bp']:
                for marker_filter in ['all-markers', 'significant', 'balanced']:
                    for atlas_with_rep in [True, False]:
                        for n_unknowns in [0, 1]:
                            process(method, experiment, markers, marker_filter, atlas_with_rep=atlas_with_rep, n_unknowns=n_unknowns, override_=True)

    #for method in ['nnls', 'qp']:
    #    for experiment in ['too-cbc', 'Br62', 'Br66', 'Cer77', 'Cer81', 'Colo', 'Colo9345', 'Ov79', 'Ov9433', 'all-insilico', 'cfdna']:
    #        for markers in ['30_250bp']:
    #            for marker_filter in ['all-markers', 'significant', 'balanced']:
    #                for atlas_with_rep in [True, False]:
    #                    for n_unknowns in [0]:
    #                        process(method, experiment, markers, marker_filter, atlas_with_rep=atlas_with_rep, n_unknowns=n_unknowns, override_=True)

    #for method in ['celfie']:
    #    for experiment in ['all-insilico']:
    #        for markers in ['30_50bp', '30_100bp', '30_250bp']:
    #            for marker_filter in ['significant', 'balanced']:
    #                for atlas_with_rep in [True, False]:
    #                    for n_unknowns in [10]:
    #                        process(method, experiment, markers, marker_filter, atlas_with_rep=atlas_with_rep, n_unknowns=n_unknowns, override_=True)

    #for experiment in ['too-cbc', 'Br62', 'Br66', 'Cer77', 'Cer81', 'Colo', 'Colo9345', 'Ov79', 'Ov9433']:
    #    for markers in ['none', '30_50bp', '30_100bp']:
    #        for marker_filter in ['all-markers', 'balanced', 'significant']:
    #            for atlas_with_rep in [True, False]:
    #                for n_unknowns in [0, 1, 2]:
    #                    process('celfie', experiment, markers, marker_filter, atlas_with_rep=atlas_with_rep, n_unknowns=n_unknowns)
