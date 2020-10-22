import csv
import glob
import imageio
import os
import re
import tqdm

import xml.etree.ElementTree as ET


REPORTS_ROOT = '/data/datasets/chest_xray/NLMCXR/NLMCXR_reports'
IMAGE_ROOT = '/data/datasets/chest_xray/NLMCXR/NLMCXR_png'

reports = glob.glob(os.path.join(REPORTS_ROOT, '*.xml'))

frontal = [line.rstrip('\n') for line in open('view_frontal.txt')]
lateral = [line.rstrip('\n') for line in open('view_lateral.txt')]

reports.sort(key=lambda f: int(re.sub('\D', '', f)))

myfile = open('metadata.csv', 'w')
csvWriter = csv.writer(myfile)
csvWriter.writerow(['dicom_id',
                    'subject_id',
                    'ViewPosition',
                    'Rows',
                    'Columns',
                    'PerformedProcedureStepDescription',
                    # 'IMPRESSION',
                    'FINDINGS'])
                    # 'REPORT'])


count = 0

for idx, report in tqdm.tqdm(enumerate(reports), total=len(reports)):

    tree = ET.parse(report)
    root = tree.getroot()

    for child in root:
        if child.tag == 'parentImage':
            # print(child.attrib['id'])

            view_pos = []
            image = []

            _frontal = [s for s in frontal if child.attrib['id'] in s]
            _lateral = [s for s in lateral if child.attrib['id'] in s]

            findings_text = root[16][0][2][2].text
            impression_text = root[16][0][2][3].text

            if _frontal:
                count = count + 1
                view_pos = 'frontal'
                image = imageio.imread(os.path.join(IMAGE_ROOT, _frontal[0]))
            elif _lateral:
                count = count + 1
                view_pos = 'lateral'
                image = imageio.imread(os.path.join(IMAGE_ROOT, _lateral[0]))
            else:
                print('ERR!')

            subject_id = child.attrib['id'].split('_')[0].replace('CXR', '')
            dicom_id = child.attrib['id']
            desc = child[1].text
            Y = image.shape[0]
            X = image.shape[1]

            # print(dicom_id, subject_id, view_pos, Y, X, desc)
            csvWriter.writerow([dicom_id,
                                subject_id,
                                view_pos,
                                Y,
                                X,
                                desc,
                                # impression_text,
                                findings_text])
            # ' '.join(filter(None, [findings_text, impression_text]))])

myfile.close()
print(f'Done! {count} images processed.')
