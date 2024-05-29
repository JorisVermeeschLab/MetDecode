import os

import numpy as np

from metdecode.io import load_input_file, save_counts


ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT, '..', 'data')


#for atlas_modifier in ['30_50bp', '30_100bp', '30_250bp']:
for atlas_modifier in ['30_250bp']:
    for markers in ['all-markers', 'balanced', 'significant']:
        print(atlas_modifier, markers)

        input_files = [
            'cfdna.tsv',
            'gdna-for-cbc-and-too.tsv',
            'insilico_Br62.tsv',
            'insilico_Br66.tsv',
            'insilico_Cer77.tsv',
            'insilico_Cer81.tsv',
            'insilico_Colo.tsv',
            'insilico_Colo9345.tsv',
            'insilico_Ov79.tsv',
            'insilico_Ov9433.tsv'
        ]

        all_M = []
        all_D = []
        all_samples_names = []
        for input_file in input_files:
            filepath = os.path.join(INPUT_DIR, atlas_modifier, markers, input_file)
            M, D, sample_names, marker_names = load_input_file(filepath)
            all_M.append(M)
            all_D.append(D)
            all_samples_names.append(sample_names)
        all_M = np.concatenate(all_M, axis=0).astype(int)
        all_D = np.concatenate(all_D, axis=0).astype(int)
        all_samples_names = np.concatenate(all_samples_names, axis=0)

        out_filepath = filepath = os.path.join(INPUT_DIR, atlas_modifier, markers, 'all.tsv')
        save_counts(out_filepath, all_M, all_D, all_samples_names, marker_names)
