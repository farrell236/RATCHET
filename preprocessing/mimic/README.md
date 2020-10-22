# MIMIC-CXR v2.0.0 Dataset


## Download dataset

Main Project Website: https://mimic-cxr.mit.edu

- DICOM images: [Link](https://physionet.org/content/mimic-cxr/2.0.0/)
- JPG images: [Link](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)


## Files

Code in this folder pre-processes the mimic-cxr dataset for text generation and Streamlit demo tasks featured in this repository. There should be <ins>6</ins> additional supporting files in this folder.


##### &gt;&gt; `mimic_cxr_labeled.csv`

csv of the entire MIMIC-CXR v2.0.0 dataset, with fields for; CXR Image Study ID, Radiological Text Report, and 14 pathology class labels via CheXpert Labeler. This file is created by code in the repository [`farrell/mimic-cxr`](https://github.com/farrell236/mimic-cxr/) (forked from [`MIT-LCP/mimic-cxr`](https://github.com/MIT-LCP/mimic-cxr/)), and contains additional fixes for the radiological texts extraction process.

1. Run [`create_section_files.py`](https://github.com/farrell236/mimic-cxr/blob/master/txt/create_section_files.py) to extract text from each report.
2. Label the reports using CheXpert Labeler (instructions [here](https://github.com/farrell236/mimic-cxr/blob/master/txt/chexpert/README.md))
3. Run [`merge_sections.py`](https://github.com/farrell236/mimic-cxr/blob/master/txt/merge_sections.py) to merge the sections together into a single csv.

Note: CheXpert Labeler is single core only, and it stores all records in memory before writing out the csv. It is advised to use the split/merge method as it can be accelerated by [GNU Parallel](https://www.gnu.org/software/parallel/). See: [`run_chexpert_on_files_parallel.sh`](https://github.com/farrell236/mimic-cxr/blob/master/txt/chexpert/run_chexpert_on_files_parallel.sh).


##### &gt;&gt; `MIMIC_AP_PA_train.csv`, `MIMIC_AP_PA_validate.csv` and `MIMIC_AP_PA_test.csv`:

These are created by the script [`create_dataset_csv.py`](https://github.com/farrell236/RATCHET/blob/master/preprocessing/mimic/create_dataset_csv.py). The script reads `mimic_cxr_labeled.csv` and splits the data according to the [original split](https://physionet.org/content/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz). Additionally, only AP or PA views are used.


##### &gt;&gt; `mimic-merges.txt` and `mimic-vocab.json`:

These are the vocab dictionary files, created by [`create_bpe_vocab.py`](https://github.com/farrell236/RATCHET/blob/master/preprocessing/mimic/create_bpe_vocab.py). The files are used by [Huggingface Byte-Level BPE Tokenizer](https://github.com/huggingface/tokenizers) in the tokenization process.


## Note
Support files are **not** included in this repository as user credentialing is required for data access. See: [here](https://mimic-cxr.mit.edu/about/access/)
