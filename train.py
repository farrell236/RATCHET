import os
import torch
import tqdm

import numpy as np

from datasets.mimic import CustomImageDataset
from model.transformer import Transformer
from model.lr_scheduler import CustomSchedule
from model.utils import create_target_masks

from tokenizers import ByteLevelBPETokenizer


'''
STEP 1: LOAD DATASET
'''
csv_root = 'preprocessing/mimic'
img_dir = '/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org'

# Parameters
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Generators
training_set = CustomImageDataset(csv_root, img_dir, mode='train')
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = CustomImageDataset(csv_root, img_dir, mode='validate')
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

tokenizer = ByteLevelBPETokenizer(
    os.path.join(csv_root, 'mimic-vocab.json'),
    os.path.join(csv_root, 'mimic-merges.txt'),
)

'''
STEP 4: INSTANTIATE MODEL
'''
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1
model_name = 'transformer_model_v2'

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    target_vocab_size=tokenizer.get_vocab_size(),
    rate=dropout_rate,
    device=torch.device('cuda')
)


# transformer = torch.load('transformer_model.pt')


'''
STEP 5: INSTANTIATE LOSS
'''
loss_object = torch.nn.CrossEntropyLoss(reduction='none')


def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, 0))
    loss_ = loss_object(pred.permute([0, 2, 1]), real.long())

    mask = mask.type(dtype=loss_.dtype)
    loss_ *= mask

    return torch.sum(loss_)/torch.sum(mask)


def accuracy_function(real, pred):
    accuracies = torch.eq(real, torch.argmax(pred, dim=2))

    mask = torch.logical_not(torch.eq(real, 0))
    accuracies = torch.logical_and(mask, accuracies)

    accuracies = accuracies.type(dtype=torch.float32)
    mask = mask.type(dtype=torch.float32)
    return torch.sum(accuracies)/torch.sum(mask)


'''
STEP 6: INSTANTIATE OPTIMIZER
'''
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = CustomSchedule(d_model, warmup_steps=8000)


'''
STEP 7: TRAIN THE MODEL
'''
global_step = 1
for epoch in range(max_epochs):  # loop over the dataset multiple times

    total_loss = 0

    t = tqdm.tqdm(enumerate(training_generator), total=len(training_generator))
    for (i, (inp, tar)) in t:

        # Fetch text from dataset iterator
        inp = torch.cat(3*[inp], dim=1)
        inp = inp.to(device=torch.device('cuda'))
        tar = tar.to(device=torch.device('cuda'))

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        # Create padding masks for input/target
        with torch.no_grad():
            combined_mask = create_target_masks(tar_inp)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # adjust learning rate for training step
        current_lr = 0  # scheduler.adjust_learning_rate(optimizer, global_step)

        # Forward pass to get output/logits
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     combined_mask,
                                     None)

        # Calculate Sparse Categorical Loss
        loss = loss_function(tar_real, predictions)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        # Updating step parameters
        total_loss += loss.detach().item()
        global_step += 1
        t.set_description(f'>> Global Step {global_step}: '
                          f'loss={loss.detach().item():.5f}, '
                          f'lr={current_lr:.5f}')

    torch.save(transformer, model_name+'.pt')
    print(f'Epoch {epoch}: avg_step_loss={total_loss / len(training_generator):.5f}')

print('Finished Training')
