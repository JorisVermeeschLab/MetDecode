import copy
import random
import enum
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import uuid
import sys
import tqdm
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.patches as mpatches

from metdecode.evaluation import Evaluation
from metdecode.io import load_input_file
from metdecode.model import MetDecode


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_FOLDER = os.path.join(ROOT, 'data')
ATLAS_FILEPATH = os.path.join(DATA_FOLDER, '30_50bp', 'balanced', 'atlas.tsv')
CFDNA_FILEPATH = os.path.join(DATA_FOLDER, '30_50bp', 'balanced', 'cfdna.tsv')
OUT_DIR = os.path.join(ROOT, 'figures')


def generate_unknown(R: np.ndarray) -> np.ndarray:
    return np.asarray([random.choice(x) for x in R.T])


def generate_mix_dataset(
        n_profiles: int = 500,
        n_unknown_tissues: int = 0
) -> dict:

    # Load reference atlas
    R_methylated, R_depths, cell_type_names, row_names = load_input_file(ATLAS_FILEPATH)
    n_known_tissues = len(R_methylated)
    n_tissues = n_known_tissues + n_unknown_tissues
    n_markers = len(row_names)
    row_names = np.asarray(row_names, dtype=object)
    assert len(cell_type_names) == len(R_methylated)
    assert len(cell_type_names) == len(R_depths)
    assert R_methylated.shape[0] == n_known_tissues
    assert n_known_tissues <= n_tissues
    assert n_markers <= R_depths.shape[1]

    # Load cfDNA samples to get an overall coverage profile
    _, X_depths, _, _ = load_input_file(CFDNA_FILEPATH)
    x_depths = np.mean(X_depths, axis=0)
    X_depths = np.clip(scipy.stats.poisson.rvs(x_depths, size=(n_profiles, n_markers)), 1, None)

    marker_idx = np.asarray([10, 11, 4, 2, 7, 8, 3, 1, 5, 0, 12, 6, 9], dtype=int)
    X_depths = X_depths.reshape(len(X_depths), 13, -1)[:, marker_idx, :].reshape(len(X_depths), -1)
    R_methylated = R_methylated.reshape(len(R_methylated), 13, -1)[:, marker_idx, :].reshape(len(R_methylated), -1)
    R_depths = R_depths.reshape(len(R_depths), 13, -1)[:, marker_idx, :].reshape(len(R_depths), -1)

    # Compute beta values
    R_methylated, R_depths = MetDecode.add_pseudo_counts(R_methylated, R_depths)
    gamma = R_methylated / R_depths

    # Add unknown cell types to the atlas
    for i in range(n_unknown_tissues):
        unk = generate_unknown(gamma)
        gamma = np.concatenate((gamma, unk[np.newaxis, :]), axis=0)
        R_depths = np.concatenate((R_depths, np.median(R_depths, axis=0)[np.newaxis, :]), axis=0)

    # Random sampling of cell type proportions
    alpha = np.zeros((n_profiles, n_tissues))
    # prior = [0.7393, 3.0938, 8.2576, 3.8222, 1.7946, 4.6937, 2.6914, 29.6514]
    prior = [5, 3.0938, 8.2576, 3.8222, 1.7946, 4.6937, 2.6914, 29.6514]
    for i in range(n_unknown_tissues):
        prior.append(3)
    for i in range(len(alpha)):
        alpha_ = np.random.dirichlet(np.asarray(prior))
        j = np.random.randint(0, 6)
        alpha[i, j] = alpha_[0]
        alpha[i, 6:] = alpha_[1:]
    cancer_mask = np.zeros(n_tissues, dtype=bool)
    cancer_mask[:6] = True

    # Compute beta values of cfDNA samples
    X_gamma = np.dot(alpha, gamma)

    # Random sampling of the methylated counts
    coverage_factor = 1
    R_depths = np.round(R_depths * coverage_factor).astype(int)
    X_depths = np.round(X_depths * coverage_factor).astype(int)

    #R_methylated = np.round(R_depths * gamma).astype(int)
    #X_methylated = np.round(X_depths * X_gamma).astype(int)
    R_methylated = np.random.binomial(R_depths, gamma)
    X_methylated = np.random.binomial(X_depths, X_gamma)

    R_methylated = np.clip(R_methylated, 0, R_depths)
    X_methylated = np.clip(X_methylated, 0, X_depths)
    assert R_methylated.shape[1] == n_markers
    assert R_depths.shape[1] == n_markers
    assert alpha.shape == (n_profiles, n_tissues)
    assert gamma.shape == (n_tissues, n_markers)
    assert X_methylated.shape == (n_profiles, n_markers)
    assert X_depths.shape == (n_profiles, n_markers)
    assert R_methylated.shape == (n_tissues, n_markers)
    assert R_depths.shape == (n_tissues, n_markers)
    R_methylated, R_depths = MetDecode.add_pseudo_counts(R_methylated, R_depths)
    X_methylated, X_depths = MetDecode.add_pseudo_counts(X_methylated, X_depths)

    R_depths = R_depths[:n_known_tissues, :]
    R_methylated = R_methylated[:n_known_tissues, :]

    return {
        'X-methylated': X_methylated,
        'X-depths': X_depths,
        'R-methylated': R_methylated[:n_known_tissues],
        'R-depths': R_depths[:n_known_tissues],
        'Alpha': alpha,
        'Gamma': gamma[:n_known_tissues],
        'n-known-tissues': n_known_tissues,
        'n-unknown-tissues': n_tissues - n_known_tissues,
        'cancer-mask': cancer_mask,
        'row-names': row_names
    }


def evaluate(dataset: dict, evaluation: Evaluation, exp_id: str, n_unknown_tissues: int = 0):
    n_known_tissues = dataset['R-methylated'].shape[0]

    args = [
        dataset['R-methylated'],
        dataset['R-depths'],
        dataset['X-methylated'],
        dataset['X-depths'],
    ]
    M_atlas, D_atlas, M_cfdna, D_cfdna = args
    gamma_hat = M_atlas / D_atlas
    
    R_atlas = M_atlas / D_atlas
    R_cfdna = M_cfdna / D_cfdna
    alpha = []
    for i in range(M_cfdna.shape[0]):
        x, residuals = scipy.optimize.nnls(R_atlas.T, R_cfdna[i, :])
        alpha.append(x / np.sum(x))
    alpha_hat = np.asarray(alpha)
    evaluation.add(alpha_hat, dataset['Alpha'], gamma_hat, dataset['Gamma'],
        dataset['X-methylated'], dataset['X-depths'], 'nnls', exp_id, n_known_tissues)


def run_simulation():

    base_scores = []
    all_scores = []

    for _ in tqdm.tqdm(range(100)):

        cell_types = [
            'Erythroblast', #0
            'CD4+ T cell', #1
            'COAD',#2
            'B cell',#3
            'CESC',#4
            'CD8+ T cell',#5
            'NK cell',#6
            'OV',#7
            'READ',#8
            'Neutrophil',#9
            'BRCA',#10
            'CEAD',#11
            'Monocyte'#12
        ]
        cell_type_names = ['BRCA', 'CEAD', 'CESC', 'COAD', 'OV', 'READ', 'B cell', 'CD4+ T cell', 'CD8+ T cell', 'Erythroblast', 'Monocyte', 'NK cell', 'Neutrophil']

        dataset = generate_mix_dataset(
            n_unknown_tissues=0,
            n_profiles=1000
        )

        evaluation = Evaluation()
        evaluate(dataset, evaluation, f'sim')
        #base_scores = np.mean(np.square(evaluation.data['nnls']['sim']['alpha'] - evaluation.data['nnls']['sim']['alpha-pred']), axis=0)
        base_scores.append(evaluation.data['nnls']['sim']['pearson'])

        scores = []
        for k in range(13):

            mask = np.ones(13 * 23, dtype=bool)
            mask[23*k:23*(k+1)] = False

            sub_dataset = copy.copy(dataset)
            sub_dataset['X-methylated'] = sub_dataset['X-methylated'][:, mask]
            sub_dataset['X-depths'] = sub_dataset['X-depths'][:, mask]
            sub_dataset['R-methylated'] = sub_dataset['R-methylated'][:, mask]
            sub_dataset['R-depths'] = sub_dataset['R-depths'][:, mask]
            sub_dataset['Gamma'] = sub_dataset['Gamma'][:, mask]

            evaluation = Evaluation()
            evaluate(sub_dataset, evaluation, f'sim')

            #score = np.mean(np.square(evaluation.data['nnls']['sim']['alpha'][:, k] - evaluation.data['nnls']['sim']['alpha-pred'][:, k]))
            score = evaluation.data['nnls']['sim']['pearson'][k]
            scores.append(score)

        all_scores.append(scores)

        #for k in range(13):
        #    #print(cell_type_names[k], scipy.stats.ttest_ind(all_scores[0], all_scores[k + 1], alternative='less'))
        #    print(cell_type_names[k], base_scores[k], scores[k])

    base_scores = np.asarray(base_scores)
    all_scores = np.asarray(all_scores)

    print(base_scores.shape, all_scores.shape)

    labels = []
    def add_label(violin, label):
        color = violin['bodies'][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    plt.style.use('seaborn-dark-palette')
    plt.figure(figsize=(16, 6))
    ax = plt.subplot(1, 1, 1)
    add_label(ax.violinplot(base_scores, positions=np.arange(13) - 0.15, widths=0.25), 'All markers')
    add_label(ax.violinplot(all_scores, positions=np.arange(13) + 0.15, widths=0.25), 'Cell type removed')
    for k in range(13):
        p_value = scipy.stats.ttest_ind(all_scores[:, k], base_scores[:, k], alternative='less')[1]
        text = f'{p_value:.5f}'
        y = min(np.min(all_scores[:, k]), np.min(base_scores[:, k])) - 0.005
        ax.annotate(text, (k, y), ha='center')
    ax.set_ylabel('Pearson correlation coefficient')
    ax.set_xticks(range(len(cell_type_names)))
    ax.set_xticklabels(cell_type_names, rotation=90)
    ax.grid(linestyle='--', alpha=0.6, color='grey', linewidth=0.5)
    ax.legend(*zip(*labels), loc=3)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUT_DIR, 'lo-res', 'sim-markers-loo-1000.png'), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'hi-res', 'sim-markers-loo-1000.png'), dpi=1200)


if __name__ == '__main__':

    run_simulation()
