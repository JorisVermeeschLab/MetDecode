from typing import List
import os
import subprocess

import numpy as np


def run_cancerlocator(
    cancerlocator_location: str,
    tmp_folder: str,
    cell_type_names: List[str],
    R_atlas: np.ndarray,
    sample_names: np.ndarray,
    M_cfdna: np.ndarray,
    D_cfdna: np.ndarray,
    theta_step: float = 0.01,
    methylation_range_cutoff: float = 0.0,
    loglikelihood_ratio_cutoff: float = 0.000001,
    n_threads: int = 32
) -> str:

    for j in range(len(cell_type_names)):
        if 'BRCA' in cell_type_names[j]:
            cell_type_names[j] = 'breast'
        elif 'OV' in cell_type_names[j]:
            cell_type_names[j] = 'ovarian'
        elif ('CESC' in cell_type_names[j]) or ('CEAD' in cell_type_names[j]):
            cell_type_names[j] = 'cervical'
        elif ('READ' in cell_type_names[j]) or ('COAD' in cell_type_names[j]):
            cell_type_names[j] = 'colorectal'
        else:
            cell_type_names[j] = 'plasma_background'

    # Create temporary folder
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder)

    # Create atlas input file (methylation ratios)
    with open(os.path.join(tmp_folder, 'train'), 'w') as f:
        for j in range(len(cell_type_names)):
            f.write(cell_type_names[j] + '\t')
            f.write('\t'.join([f'{float(x):.4f}' for x in R_atlas[j, :]]))
            if j < len(cell_type_names) - 1:
                f.write('\n')

    # Create cfDNA input file (methylated CpGs)
    with open(os.path.join(tmp_folder, 'test_methy'), 'w') as f:
        for i in range(len(sample_names)):
            f.write(sample_names[i] + '\t')
            f.write('\t'.join([f'{int(x)}' for x in M_cfdna[i, :]]))
            if i  < len(sample_names) - 1:
                f.write('\n')

    # Create cfDNA input file (total CpGs)
    with open(os.path.join(tmp_folder, 'test_depth'), 'w') as f:
        for i in range(len(sample_names)):
            f.write(sample_names[i] + '\t')
            f.write('\t'.join([f'{int(x)}' if (x > 0) else 'NA' for x in D_cfdna[i, :]]))
            if i  < len(sample_names) - 1:
                f.write('\n')

    # Create file containing the (cell_type->prediction_class) mapping
    done = set()
    with open(os.path.join(tmp_folder, 'type2class'), 'w') as f:
        f.write('plasma_background\tnormal\n')
        f.write('breast\tbreast cancer\n')
        f.write('cervical\tcervical cancer\n')
        f.write('colorectal\tcolorectal cancer\n')
        f.write('ovarian\tovarian cancer')

    # Create CancerLocator configuration file
    with open(os.path.join(tmp_folder, 'config'), 'w') as f:
        f.write(f'trainFile={tmp_folder}/train\n')
        f.write(f'testMethyFile={tmp_folder}/test_methy\n')
        f.write(f'testDepthFile={tmp_folder}/test_depth\n')
        f.write(f'typeMappingFile={tmp_folder}/type2class\n')
        f.write(f'resultFile={tmp_folder}/result\n')
        f.write(f'thetaStep={theta_step}\n')
        f.write(f'methylationRangeCutoff={methylation_range_cutoff}\n')
        f.write(f'logLikelihoodRatioCutoff={loglikelihood_ratio_cutoff}\n')
        f.write(f'nThreads={n_threads}')

    # Run CancerLocator
    subprocess.run(['java', '-jar', cancerlocator_location, os.path.join(tmp_folder, 'config')])

    return os.path.join(tmp_folder, 'result')
