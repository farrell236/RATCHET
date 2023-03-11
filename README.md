# RATCHET: RAdiological Text Captioning for Human Examined Thoraxes

RATCHET is a Medical Transformer for Chest X-ray Diagnosis and Reporting. Based on the architecture featured in [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). This network is trained and validated on the MIMIC-CXR v2.0.0 dataset.


### Architecture

<p align="center">
  <img src="assets/model_transformer.png" alt="RATCHET Architecture" width="300"/>
</p>


### Run the code

Download pretrained weights and put in `./checkpoints` folder.

- [ratchet_model_weights_202303111506.zip](http://www.doc.ic.ac.uk/~bh1511/ratchet_model_weights_202303111506.zip)
<br> Size: `1.5G` <br> MD5: `26ab19cf18908841320205e192dabe9f` <br>

Start streamlit to run the webapp:

```
streamlit run web_demo.py
```

##### Environment: 
```
Python 3.9.10
```

##### Packages:
```
imageio                  2.26.0
matplotlib               3.7.1
numpy                    1.23.5
pandas                   1.5.3
scikit-image             0.20.0
streamlit                1.20.0
tensorflow               2.11.0
tokenizers               0.13.2
tqdm                     4.64.1
```


### Docker Container

Build the docker container:

```
docker build -t ratchet ./Dockerfile
```

Run the docker image on CXR images:

```
docker run --user $(id -u):$(id -g) \
-v /path/to/image_input_folder:/code/RATCHET/inp_folder \
-v /path/to/report_output_folder:/code/RATCHET/out_folder:rw \
-i -t ratchet python run_model.py
```

Each image in `inp_folder` would have a corresponding `.txt` report saved in `out_folder`.


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
