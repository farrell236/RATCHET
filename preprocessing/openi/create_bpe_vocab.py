import pandas as pd

from tokenizers import ByteLevelBPETokenizer

OPENI_REPORTS = 'metadata_dataset.csv'

reports = pd.read_csv(OPENI_REPORTS)
reports = reports.dropna(subset=['FINDINGS'])  # delete empty reports
reports = list(reports['FINDINGS'].values)

with open('/tmp/openi.txt', 'w') as f:
    for item in reports:
        f.write("%s\n" % item)

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files='/tmp/openi.txt', vocab_size=20000, min_frequency=2, special_tokens=[
    '<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',
])

tokenizer.save('.', 'openi')
