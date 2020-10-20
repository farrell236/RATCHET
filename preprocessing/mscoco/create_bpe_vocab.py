import json
import os
import tqdm

from tokenizers import ByteLevelBPETokenizer

captions_file = '/data/datasets/MS-COCO/2017/annotations/captions_train2017.json'

# Store captions and image names in vectors
all_captions = []

# Read the annotation json file
with open(os.path.join(captions_file), 'r') as f:
    annotations = json.load(f)

for annot in tqdm.tqdm(annotations['annotations']):
    all_captions.append(annot['caption'])

# Write all captions to txt file
print('Writing captions to /tmp/coco_captions.txt')
with open('/tmp/coco_captions.txt', 'w') as f:
    for item in all_captions:
        f.write("%s\n" % item)

# Create BPE Tokenizer and train on coco captions
print('Training BPE Tokenizer')
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files='/tmp/coco_captions.txt', vocab_size=52000, min_frequency=2, special_tokens=[
    '<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',
])

# Save vocab dictionary
tokenizer.save('.', 'coco')
