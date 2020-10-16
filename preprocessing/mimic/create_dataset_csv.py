import os
import tqdm

import pandas as pd


MIMIC_REPORTS   = './mimic_cxr_labeled.csv'
mimic_root      = '/path/to/mimic-cxr-jpg-2.0.0.physionet.org/'

# Read CSVs into pandas dataframes
split_df        = pd.read_csv(os.path.join(mimic_root, 'mimic-cxr-2.0.0-split.csv.gz'))
metadata_df     = pd.read_csv(os.path.join(mimic_root, 'mimic-cxr-2.0.0-metadata.csv.gz'))
reports_df      = pd.read_csv(MIMIC_REPORTS)

# Combine dataset splitting with metadata
combined = pd.concat((split_df, metadata_df.iloc[:, 3:]), axis=1)

# Use PA or AP view position only
combined = combined.loc[combined['ViewPosition'].isin(['PA', 'AP'])]

# No reports
missing_studies = [
    58235663,
    50798377,
    54168089,
    53071062,
    56724958,
    54231141,
    53607029,
    52035334,
]


for mode in ['train', 'validate', 'test']:

    print(f'Processing Dataset: \'{mode}\'')

    df = combined.loc[combined['split'] == mode]
    records = []

    for idx in tqdm.tqdm(range(len(df))):
        dicom_id, study_id, subject_id, split = df.iloc[idx, :4]

        img_path = f'files/p{str(subject_id)[:2]}/p{str(subject_id)}/s{str(study_id)}/{dicom_id}.jpg'

        # Sanity check image exists
        if not os.path.exists(os.path.join(mimic_root, img_path)):
            print(f'IMAGE MISSING: {img_path}')
            continue

        if study_id in missing_studies:
            print(f'{idx}: study [{study_id}] is missing...')
            continue

        label = reports_df.loc[reports_df['Study'] == f's{study_id}']
        label.insert(0, 'DicomPath', img_path)

        records.append(label)

    records_df = pd.concat(records)
    records_df = records_df.drop(['Study'], axis=1)
    records_df.to_csv(f'MIMIC_AP_PA_{mode}.csv', index=False)
