from __future__ import print_function
from nlp_metrics.pycocotools.coco import COCO
from nlp_metrics.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

a=1

annFile = '/data/datasets/MS-COCO/2017/annotations/captions_val2017.json'
resFile = 'predictions.json'


a=1

# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

a=1

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()

a=1

for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))

a=1

'''
Bleu_1: 0.651
Bleu_2: 0.472
Bleu_3: 0.338
Bleu_4: 0.243
METEOR: 0.228
ROUGE_L: 0.489
CIDEr: 0.767
SPICE: 0.157
'''
