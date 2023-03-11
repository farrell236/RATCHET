FROM python:3.7-slim-buster
RUN pip install imageio==2.9.0 matplotlib numpy pandas scikit-image streamlit tokenizers tqdm tensorflow
WORKDIR /code
RUN cd /code
RUN apt-get update && apt-get install -y git wget unzip nano
RUN git clone https://github.com/farrell236/RATCHET.git
RUN wget -q http://www.doc.ic.ac.uk/~bh1511/ratchet_model_weights_202303111506.zip
RUN unzip -q ratchet_model_weights_202303111506.zip -d RATCHET/checkpoints
WORKDIR /code/RATCHET
RUN mkdir inp_folder out_folder
