import copy
import random
import enum
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import uuid
import sys
sys.path.insert(0, '..')

import numpy as np
import scipy.stats

from metdecode.evaluation import Evaluation
from metdecode.io import load_input_file, save_counts
from metdecode.model import MetDecode


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_FOLDER = os.path.join(ROOT, 'data')
ATLAS_FILEPATH = os.path.join(DATA_FOLDER, '30_50bp', 'balanced', 'atlas.tsv')
CFDNA_FILEPATH = os.path.join(DATA_FOLDER, '30_50bp', 'balanced', 'cfdna.tsv')

EXPERIMENT = 'variable-unk'
assert EXPERIMENT in {'no-unk', 'variable-unk', 'fixed-unk'}
OUT_FOLDER = os.path.join(ROOT, 'sim-results', EXPERIMENT)
OUT_FOLDER2 = os.path.join(ROOT, 'sim-results', 'loo')

if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)
if not os.path.exists(OUT_FOLDER2):
    os.makedirs(OUT_FOLDER2)


class ExperimentType(enum.Enum):

    UNKNOWNS = enum.auto()
    COVERAGE = enum.auto()


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


def evaluate(exp: ExperimentType, dataset: dict, evaluation: Evaluation, exp_id: str, n_unknown_tissues: int = 0):
    n_known_tissues = dataset['R-methylated'].shape[0]

    args = [
        dataset['R-methylated'],
        dataset['R-depths'],
        dataset['X-methylated'],
        dataset['X-depths'],
    ]
    M_atlas, D_atlas, M_cfdna, D_cfdna = args
    gamma_hat = M_atlas / D_atlas

    if exp == ExperimentType.UNKNOWNS:
        METHOD_NAMES = ['nnls', 'qp', 'celfie', 'metdecode', 'metdecode-nu']
    else:
        METHOD_NAMES = ['nnls', 'qp', 'celfie', 'metdecode', 'metdecode-nc']
    for method_name in METHOD_NAMES:
        if method_name == 'metdecode':
            model = MetDecode(*args, n_unknown_tissues=n_unknown_tissues, coverage=True)
            alpha_hat = model.deconvolute()
            gamma_hat = model.gamma_hat
        elif method_name == 'metdecode-nc':
            model = MetDecode(*args, n_unknown_tissues=n_unknown_tissues, coverage=False)
            alpha_hat = model.deconvolute()
            gamma_hat = model.gamma_hat
        elif method_name == 'metdecode-nu':
            model = MetDecode(*args, n_unknown_tissues=0, coverage=False)
            alpha_hat = model.deconvolute()
            gamma_hat = model.gamma_hat
        elif method_name == 'nnls':
            R_atlas = M_atlas / D_atlas
            R_cfdna = M_cfdna / D_cfdna
            alpha = []
            for i in range(M_cfdna.shape[0]):
                x, residuals = scipy.optimize.nnls(R_atlas.T, R_cfdna[i, :])
                alpha.append(x / np.sum(x))
            alpha_hat = np.asarray(alpha)
        elif method_name == 'qp':
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
            alpha_hat = alpha
        elif method_name == 'celfie':
            from metdecode.celfie import celfie_deconvolute
            alpha_hat, gamma_hat = celfie_deconvolute(M_cfdna, D_cfdna, M_atlas, D_atlas, max_n_iter=100, n_runs=1, convergence_rate=0.0001, n_unknown_tissues=n_unknown_tissues, verbose=True)
        else:
            raise NotImplementedError(f'Unknown deconvolution algorithm "{method_name}"')
        evaluation.add(alpha_hat, dataset['Alpha'], gamma_hat, dataset['Gamma'],
            dataset['X-methylated'], dataset['X-depths'], method_name, exp_id, n_known_tissues)


def run_simulation(n_unknown_tissues: int = 0):

    exp_name = f'unk_{n_unknown_tissues}'

    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    if EXPERIMENT == 'variable-unk':
        nut = n_unknown_tissues
    elif EXPERIMENT == 'no-unk':
        nut = 0
    elif EXPERIMENT == 'fixed-unk':
        nut = 100
    else:
        raise NotImplementedError()
    dataset = generate_mix_dataset(
        n_unknown_tissues=nut,
        n_profiles=100
    )
    evaluation = Evaluation()
    exp = ExperimentType.UNKNOWNS if (n_unknown_tissues > 0) else ExperimentType.COVERAGE
    evaluate(exp, dataset, evaluation, f'sim', n_unknown_tissues=n_unknown_tissues)
    out_folder = os.path.join(OUT_FOLDER, exp_name)
    os.makedirs(out_folder, exist_ok=True)
    evaluation.save(os.path.join(out_folder, f'results-{uuid.uuid4()}.pkl'))


def run_simulation2():

    dataset = generate_mix_dataset(
        n_unknown_tissues=0,
        n_profiles=1000
    )

    evaluation = Evaluation()

    for j in range(13):
        mask = np.ones(13, dtype=bool)
        mask[j] = False
        sub = copy.copy(dataset)
        sub['R-methylated'] = sub['R-methylated'][mask, :]
        sub['R-depths'] = sub['R-depths'][mask, :]
        sub['Alpha'] = np.concatenate((sub['Alpha'][:, mask], sub['Alpha'][:, ~mask]), axis=1)
        sub['Gamma'] = np.concatenate((sub['Gamma'][mask, :], sub['Gamma'][~mask, :]), axis=0)
        sub['cancer-mask'] = sub['cancer-mask'][mask]

        evaluate(ExperimentType.UNKNOWNS, sub, evaluation, f'{j}-removed', n_unknown_tissues=1)
    
    out_folder = OUT_FOLDER2
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    evaluation.save(os.path.join(out_folder, f'results-{uuid.uuid4()}.pkl'))


if __name__ == '__main__':

    run_simulation(n_unknown_tissues=0)

    #for _ in range(10):
    #    run_simulation2()

    #for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100]:
    #    for _ in range(10):
    #        run_simulation(n_unknown_tissues=k)

    #for k in [1]:
    #    for _ in range(190):
    #        run_simulation(n_unknown_tissues=k)
