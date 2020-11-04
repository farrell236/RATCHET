NLP Metrics 
===

Evaluation metrics for caption generation. Code in this folder are based on the [Python 3 Fork](https://github.com/flauted/coco-caption) of 
the [COCO Caption Evaluation](https://github.com/tylin/coco-caption) library, and have been modified to allow for other datasets. 

## Requirements ##
- java 1.8.0
- python (tested 2.7/3.6)

## Files ##
- eval: The file includes MIMICEavlCap and COCOEavlCap class.
- tokenizer: Python wrapper of Stanford CoreNLP PTBTokenizer
- bleu: Bleu evalutation codes
- meteor: Meteor evaluation codes
- rouge: Rouge-L evaluation codes
- cider: CIDEr evaluation codes
- spice: SPICE evaluation codes
- pycocotools and common: Adapted from [COCO Dataset API](https://github.com/cocodataset/cocoapi)

## Setup ##
- You will first need to download the [Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html) code and models for use by SPICE. To do this, run either: 
    - ``python get_stanford_models.py``
    - ``./get_stanford_models.sh``
    
- Run ``make`` in this directory to build dependencies for ``pycocotools``. Requires ``cython>=0.27.3``
    
## Notes ##
- SPICE will try to create a cache of parsed sentences in ./spice/cache/. This dramatically speeds up repeated evaluations. 
    - Without altering this code, use the environment variables ``SPICE_CACHE_DIR`` and ``SPICE_TEMP_DIR`` to set the cache directory.
    - The cache should **NOT** be on an NFS mount.
    - Caching can be disabled by editing the ``/spice/spice.py`` file. Remove the ``-cache`` argument to ``spice_cmd``.


## References ##
- [Microsoft COCO Captions: Data Collection and Evaluation Server](http://arxiv.org/abs/1504.00325)
- PTBTokenizer: We use the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) which is included in [Stanford CoreNLP 3.4.1](http://nlp.stanford.edu/software/corenlp.shtml).
- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)
- Meteor: [Project page](http://www.cs.cmu.edu/~alavie/METEOR/) with related publications. We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor). Changes have been made to the source code to properly aggreate the statistics for the entire corpus.
- Rouge-L: [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf)
- CIDEr: [CIDEr: Consensus-based Image Description Evaluation](http://arxiv.org/pdf/1411.5726.pdf)
- SPICE: [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822)

## Developers ##
- Xinlei Chen (CMU)
- Hao Fang (University of Washington)
- Tsung-Yi Lin (Cornell)
- Ramakrishna Vedantam (Virgina Tech)

## Acknowledgement ##
- David Chiang (University of Norte Dame)
- Michael Denkowski (CMU)
- Alexander Rush (Harvard University)