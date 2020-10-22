# OpenI CXR Dataset

Download Dataset
---

Main dataset page: https://openi.nlm.nih.gov/faq#collection

- PNG images: [Link](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz)
- DICOM images: [Link](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_dcm.tgz)
- Reports: [Link](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz)


Dataset Statistics
---

There are **3955 reports** for **7470 CXR images**.

- 28 records have no findings or impressions
- 6 records have only findings 
- 502 records have only impressions 
- 3419 records have findings and impressions

Of the 7470 CXR images, there are **3824 frontal** and **3646 lateral** images. See: ```view_frontal.txt``` and ```view_lateral.txt```.

9 images are corrupt and/or not in standard orientation.

CXR Image IDs:

- CXR1285_IM-0188-0001
- CXR1339_IM-0218-1001
- CXR1961_IM-0628-3001
- CXR2084_IM-0715-1001-0002
- CXR2084_IM-0715-2001-0001
- CXR2146_IM-0766-13013
- CXR2280_IM-0867-1001-0003
- CXR3809_IM-1919-1003002
- CXR994_IM-2478-1001


Dataset Images
---

All images have dimensions (x, 512)

- min(x): 362
- max(x): 873
- mean(x) 532.7789825970549
- std(x) 80.65972961332197
- distribution: 

|  Bins | 362. | 413.1 | 464.2 | 515.3 | 566.4 | 617.5 | 668.6 | 719.7 | 770.8 | 821.9 | 873. |
|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|
| Count |   -  |   13  |  1781 |  2479 |  385  |  349  |  2380 |   45  |   30  |   6   |   2  |


Mining NLP Labels
---

Run ```create_metadata_csv.py``` to generate report with following columns:


|dicom\_id|subject\_id|ViewPosition|Rows|Columns|PerformedProcedureStepDescription|FINDINGS|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| | | | | | | | |


Run [chexpert-labeler](https://github.com/stanfordmlgroup/chexpert-labeler) on the ```FINDINGS``` column to get NLP mined labels, then merge.

- metadata.csv: original output of ```create_metadata_csv.py```
- metadata_annotated.csv: output of ```create_metadata_csv.py``` with chexpert-labeler NLP mined labels
- metadata_dataset.csv: records of only frontal and without corrupt images

