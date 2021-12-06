import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import tqdm

import numpy as np
import pandas as pd

from datasets.mimic import CustomImageDataset
from model.transformer import Transformer
from model.utils import create_target_masks

from tokenizers import ByteLevelBPETokenizer


'''
STEP 1: LOAD DATASET
'''
csv_root = 'datasets'
img_dir = '/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org'

# Parameters
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}
max_epochs = 100

# Generators
test_set = CustomImageDataset(csv_root, img_dir, mode='test')
test_generator = torch.utils.data.DataLoader(test_set, **params)

tokenizer = ByteLevelBPETokenizer(
    os.path.join(csv_root, 'mimic-vocab.json'),
    os.path.join(csv_root, 'mimic-merges.txt'),
)


'''
STEP 4: INSTANTIATE MODEL
'''
transformer = torch.load('transformer_model.pt').to('cuda')
transformer.eval()

# def create_target_masks(seq):
#     return torch.eq(seq, 0).type(torch.float32)

def evaluate(inp_img, max_length=128):

    # The first token to the transformer should be the start token
    output = torch.from_numpy(np.array([[tokenizer.token_to_id('<s>')]], dtype=np.int32)).to('cuda')
    inp_img = torch.cat(3 * [inp_img], dim=1).to('cuda')

    for i in range(max_length):
        combined_mask = create_target_masks(output).to('cuda')

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(inp_img,
                                                     output,
                                                     False,
                                                     combined_mask,
                                                     None)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = torch.argmax(predictions, dim=-1)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == 2:
            break

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = torch.cat([output, predicted_id], dim=-1)

    return torch.squeeze(output, dim=0)[1:], attention_weights


pred_txt = dict()
true_txt = dict()

t = tqdm.tqdm(enumerate(test_generator), total=len(test_generator))
for (i, (inp, tar)) in t:
    true_txt[i] = tokenizer.decode(np.trim_zeros(tar[0].numpy(), 'b')[1:-1])
    result, _ = evaluate(inp)
    pred_txt[i] = tokenizer.decode(result.cpu().numpy())
    print(true_txt[i])
    print(pred_txt[i])

a=1

pred_txt_df = pd.DataFrame.from_dict(pred_txt, orient='index')
true_txt_df = pd.DataFrame.from_dict(true_txt, orient='index')

pred_txt_df.to_csv('all_pred.csv', index=False, header=False)
true_txt_df.to_csv('all_true.csv', index=False, header=False)


