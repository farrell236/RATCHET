# RATCHET: RAdiological Text Captioning for Human Examined Thoraxes

RATCHET is a Medical Transformer for Chest X-ray Diagnosis and Reporting. Based on the architecture featured in [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). This network is trained and validated on the MIMIC-CXR v2.0.0 dataset.


### Architecture

<p align="center">
  <img src="assets/model_transformer.png" alt="RATCHET Architecture" width="300"/>
</p>


### Run the code

Download the [pretrained weights](http://www.doc.ic.ac.uk/~bh1511/ratchet_model_weights_202009251103.zip) and put in `./checkpoints` folder. Then run:

```
streamlit run web_demo.py
```

##### Environment: 
```
Python 3.7.4
```

##### Packages:
```
imageio                  2.8.0
matplotlib               3.2.1
numpy                    1.18.4
pandas                   1.0.3
scikit-image             0.17.2
streamlit                0.67.1
tensorflow-gpu           2.3.0
tokenizers               0.7.0
tqdm                     4.46.0
```


### Results

<p align="center">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Cardiomegally.PNG" alt="Cardiomegaly" height="300"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/attn_plot.png" alt="Cardiomegaly Attention Plot" height="300"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
</p>


##### Generated Text: 

> In comparison with the study of \_\_\_, there is little overall change. Again there is substantial enlargement of the cardiac silhouette with a dual-channel pacer device in place. No evidence of vascular congestion or acute focal pneumonia. Blunting of the costophrenic angles is again seen.


### More Examples

<p align="center">
  <img src="assets/examples.png" alt="More Captioning Examples" max-height="400"/>
</p>
