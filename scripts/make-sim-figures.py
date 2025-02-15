import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn
import scipy.stats
from scipy.stats import ttest_rel


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_FOLDER = os.path.join(ROOT, 'sim-results', 'variable-unk')
OUT_DIR = os.path.join(ROOT, 'figures')

os.makedirs(os.path.join(OUT_DIR, 'lo-res'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'hi-res'), exist_ok=True)


def fig_a():
    for EXP1 in [True, False]:

        if EXP1:
            METHOD_NAMES = ['celfie-nu', 'celfie', 'nnls', 'qp', 'metdecode-nu', 'metdecode']
        else:
            METHOD_NAMES = ['celfie', 'nnls', 'qp', 'metdecode-nc', 'metdecode']

        res = []
        if EXP1:
            folder = 'unk_1'
        else:
            folder = 'unk_0'
        for filename in os.listdir(os.path.join(OUT_FOLDER, folder)):

            if not (filename.startswith('results-') and filename.endswith('.pkl')):
                continue
            filepath = os.path.join(OUT_FOLDER, folder, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            data = data[0]
            res.append([data[method_name]['sim']['pearson'] for method_name in METHOD_NAMES])
        res = np.asarray(res)

        if EXP1:
            colors = ['darkblue', 'royalblue', 'darkslateblue', 'slateblue', 'mediumvioletred', 'palevioletred', 'steelblue', 'darkturquoise', 'darkcyan', 'mediumseagreen', 'darkgreen', 'green', 'yellowgreen', 'tan']
        else:
            #colors = ['darkblue', 'slateblue', 'palevioletred', 'steelblue', 'mediumseagreen', 'yellowgreen', 'tan']
            colors = ['darkblue', 'darkslateblue', 'slateblue', 'mediumvioletred', 'palevioletred', 'steelblue', 'darkturquoise', 'darkcyan', 'mediumseagreen', 'darkgreen', 'green', 'yellowgreen', 'tan']
        pretty_names = [
            'BRCA', 'CEAD', 'CESC', 'COAD', 'OV', 'READ', 'B cell', 'CD4+ T-cell', 'CD8+ T-cell',
            'Erythroblast', 'Monocyte', 'Natural killer cell', 'Neutrophil', 'Average']
        plt.figure(figsize=(12.5, 7.5))
        for k in range(15):
            ax = plt.subplot(3, 5, k + 1)

            if k == 14:
                plt.axis('off')
                ys = list(np.mean(res, axis=2).T)
            elif k == 13:
                ys = list(np.mean(res, axis=2).T)
            else:
                ys = list(res[:, :, k].T)

            p_value = float(np.max([ttest_rel(ys[-1], ys[i], alternative='greater').pvalue for i in range(0, len(ys) - 1)]))

            #r = plt.violinplot(ys, positions=range(1, len(ys) + 1), showmeans=True, showextrema=True)
            #r['cbars'].set_colors(colors[1:len(ys)+1])
            #r['cmins'].set_colors(colors[1:len(ys)+1])
            #r['cmaxes'].set_colors(colors[1:len(ys)+1])
            #r['cmeans'].set_colors(colors[1:len(ys)+1])
            #for k2, body in enumerate(r['bodies']):
            #    body.set_color(colors[k2 + 1])
            #plt.grid(alpha=0.4, color='grey', linewidth=0.5, linestyle='--')
            plt.xticks([], [])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if k < 14:
                r = ax.violinplot(ys, showmeans=True, showextrema=True)
                r['cbars'].set_colors(colors[:len(ys)])
                r['cmins'].set_colors(colors[:len(ys)])
                r['cmaxes'].set_colors(colors[:len(ys)])
                r['cmeans'].set_colors(colors[:len(ys)])
                for k2, body in enumerate(r['bodies']):
                    body.set_color(colors[k2])
                plt.title(f'{pretty_names[k]} ({p_value:.3f})')
                plt.grid(alpha=0.4, color='grey', linewidth=0.5, linestyle='--')
                #plt.xticks(range(1, len(ys) + 1), [''] * len(ys))
                for side in ['right', 'top', 'bottom']:
                    ax.spines[side].set_visible(False)
                ax.set_yticks(list(ax.get_yticks()) + [1])
                ax.set_ylim([None, 1])
            else:
                if EXP1:
                    plt.legend(handles=[
                        mpatches.Patch(color=colors[0], label='CelFIe (unk=0)'),
                        mpatches.Patch(color=colors[1], label='CelFIe (unk=1)'),
                        mpatches.Patch(color=colors[2], label='NNLS'),
                        mpatches.Patch(color=colors[3], label='QP'),
                        mpatches.Patch(color=colors[4], label='MetDecode (unk=0)'),
                        mpatches.Patch(color=colors[5], label='MetDecode (unk=1)'),
                    ])
                else:
                    plt.legend(handles=[
                        mpatches.Patch(color=colors[0], label='CelFIe'),
                        mpatches.Patch(color=colors[1], label='NNLS'),
                        mpatches.Patch(color=colors[2], label='QP'),
                        mpatches.Patch(color=colors[3], label='MetDecode (no coverage)'),
                        mpatches.Patch(color=colors[4], label='MetDecode'),
                    ])
                plt.axis('off')
                continue

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'lo-res', f'sim-unk1.png' if EXP1 else f'sim-coverage.png'), dpi=1200)
        plt.savefig(os.path.join(OUT_DIR, 'hi-res', f'sim-unk1.png' if EXP1 else f'sim-coverage.png'), dpi=1200)
        plt.close()
        plt.clf()
        plt.cla()


def fig_b():

    plt.figure(figsize=(16, 6))

    k2 = 1

    for k3 in range(2):

        ax = plt.subplot(1, 2, k3 + 1)

        METHOD_NAMES = ['celfie', 'metdecode']
        COLORS = ['darkslateblue', 'tan']

        metric = 'mse' if (k3 == 1) else 'pearson'

        results = []
        xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100]
        for n_unk in xs:

            if k2 == 1:
                folder = os.path.join(ROOT, 'sim-results', 'variable-unk', f'unk_{n_unk}')
            else:
                folder = os.path.join(ROOT, 'sim-results', 'no-unk', f'unk_{n_unk}')

            res, sigma = [], []
            for filename in os.listdir(folder)[:10]:

                if not (filename.startswith('results-') and filename.endswith('.pkl')):
                    continue
                filepath = os.path.join(folder, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                data = data[0]

                row = []
                for method_name in METHOD_NAMES:

                    alpha = data[method_name]['sim']['alpha'][:, :13]
                    alpha = alpha / np.sum(alpha, axis=1)[:, np.newaxis]
                    alpha_pred = data[method_name]['sim']['alpha-pred'][:, :13]
                    alpha_pred = alpha_pred[:, :alpha.shape[1]]
                    alpha_pred = alpha_pred / np.sum(alpha_pred, axis=1)[:, np.newaxis]

                    if metric == 'mse':
                        row.append(np.mean(np.square(alpha - alpha_pred)))
                    else:
                        row.append(np.mean([scipy.stats.pearsonr(alpha[:, j], alpha_pred[:, j])[0] for j in range(alpha_pred.shape[1])]))
                    print(alpha.shape, alpha_pred.shape, np.asarray(row).shape)

                res.append(row)

            res = np.asarray(res)
            results.append(res)
        results = np.asarray(results)  # shape: (n_unk, n_repeats, n_methods)

        for k in range(2):
            mu = np.mean(results[:, :, k], axis=1)
            sigma = np.std(results[:, :, k], axis=1)
            lb = np.min(results[:, :, k], axis=1)
            ub = np.max(results[:, :, k], axis=1)
            ax.errorbar(np.arange(len(xs)), mu, yerr=sigma, label=METHOD_NAMES[k], color=COLORS[k])
            ax.fill_between(np.arange(len(xs)), lb, ub, alpha=0.5, color=COLORS[k])
        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticklabels([str(x) for x in xs])
        #ax.set_title(f'{metric} - {"variable-unk" if (k2 == 1) else "no-unk"}')
        #ax.set_xscale('log')
        ax.grid(linestyle='--', alpha=0.6, color='grey', linewidth=0.5)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel('Number of unknowns')
        if k3 == 1:
            ax.set_ylabel('Average mean squared error')
        else:
            ax.set_ylabel('Average Pearson correlation')
        if k2 == 1:
            ax.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, 'lo-res', f'sim-variable-unk.png'), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'hi-res', f'sim-variable-unk.png'), dpi=1200)

    plt.show()


def fig_c():

    cell_type_names = ['BRCA', 'CEAD', 'CESC', 'COAD', 'OV', 'READ', 'B cell', 'CD4+ T cell', 'CD8+ T cell', 'Erythroblast', 'Monocyte', 'NK cell', 'Neutrophil']

    folder = os.path.join(ROOT, 'sim-results', 'loo')

    data_cell_types = []
    data_method = []
    data_values = []

    for k, method_name in enumerate(['celfie', 'metdecode']):

        for i, filename in enumerate(os.listdir(folder)):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            for j in range(13):
                mask = np.ones(13, dtype=bool)
                mask[j] = False
                gamma = data[0][method_name][f'{j}-removed']['gamma'][mask, :]
                gamma_unk = data[0][method_name][f'{j}-removed']['gamma'][j, :]
                gamma_pred = data[0][method_name][f'{j}-removed']['gamma-pred']
                gamma_unk_pred = data[0][method_name][f'{j}-removed']['gamma-pred'][-1, :]

                data_method.append(method_name)
                data_cell_types.append(j)
                data_values.append(scipy.stats.pearsonr(gamma_unk, gamma_unk_pred)[0])
                #data_values.append(np.mean(np.square(gamma_unk - gamma_unk_pred)))

    df = pd.DataFrame({
        'cell_types': data_cell_types,
        'Method': data_method,
        'values': data_values
    })

    plt.figure(figsize=(16, 6))
    ax = plt.subplot(1, 1, 1)
    seaborn.swarmplot(ax=ax, data=df, x='cell_types', y='values', hue='Method', palette=['darkslateblue', 'tan'])
    ax.set_xticklabels(cell_type_names, rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('Pearson correlation for unknown cell type')
    ax.grid(linestyle='--', alpha=0.6, color='grey', linewidth=0.5)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, 'lo-res', f'sim-loo.png'), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'hi-res', f'sim-loo.png'), dpi=1200)


if __name__ == "__main__":
    
    fig_a()
    fig_b()
    fig_c()
