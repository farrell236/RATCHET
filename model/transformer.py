from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torchvision

import numpy as np

from .utils import positional_encoding


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = float(k.shape[-1])
    scaled_attention_logits = matmul_qk / dk ** 0.5

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=d_model, out_features=dff),
        torch.nn.LeakyReLU(negative_slope=0.2),
        torch.nn.Linear(in_features=dff, out_features=d_model)
    )


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, device=torch.device('cpu')):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = self.d_model // self.num_heads

        self.wq = torch.nn.Linear(d_model, d_model).to(device=device)
        self.wk = torch.nn.Linear(d_model, d_model).to(device=device)
        self.wv = torch.nn.Linear(d_model, d_model).to(device=device)

        self.dense = torch.nn.Linear(d_model, d_model).to(device=device)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.permute(dims=[0, 2, 1, 3])

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = \
            scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(dims=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = torch.reshape(scaled_attention,
                                         (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1, device=torch.device('cpu')):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, device=device)
        self.mha2 = MultiHeadAttention(d_model, num_heads, device=device)

        self.ffn = point_wise_feed_forward_network(d_model, dff).to(device=device)

        self.layernorm1 = torch.nn.LayerNorm(d_model).to(device=device)
        self.layernorm2 = torch.nn.LayerNorm(d_model).to(device=device)
        self.layernorm3 = torch.nn.LayerNorm(d_model).to(device=device)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(torch.nn.Module):
    def __init__(self, embedding_dim, device=torch.device('cpu')):
        super(Encoder, self).__init__()

        # Use DenseNet-121 as feature extraction model
        self.base_model = torchvision.models.densenet121().to(device=device)
        self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-1]))

        # shape after fc == (batch_size, nf * nf, embedding_dim)
        self.fc = torch.nn.Linear(in_features=1024, out_features=embedding_dim).to(device=device)

    def forward(self, x):
        x = self.base_model(x)
        # DenseNet-121 output is (batch_size, 1024, ?, ?)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, device=torch.device('cpu')):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(target_vocab_size, d_model).to(device=device)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model).to(device=device)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, device)
                           for _ in range(num_layers)]
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = x.shape[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= self.d_model ** 0.5
        x += torch.as_tensor(self.pos_encoding[:, :seq_len, :], device=x.device)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, rate=0.1, device=torch.device('cpu')):
        super(Transformer, self).__init__()

        self.encoder = Encoder(d_model, device)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, target_vocab_size, rate, device)

        self.final_layer = torch.nn.Linear(d_model, target_vocab_size).to(device=device)

    def forward(self, inp, tar, training, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


# if __name__ == '__main__':
#
#     a=1
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # dev = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     sample_transformer = Transformer(
#         num_layers=2, d_model=512, num_heads=8, dff=2048,
#         target_vocab_size=8000, device=device)
#
#     temp_input = torch.from_numpy(np.random.rand(10, 3, 224, 224).astype('float32')).to('cuda')
#     temp_target = torch.from_numpy(np.random.randint(low=0, high=200, size=(10, 36), dtype='int64')).to(device=device)
#
#     fn_out, _ = sample_transformer(temp_input, temp_target,
#                                    training=True,
#                                    look_ahead_mask=None,
#                                    dec_padding_mask=None)
#
#     fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)
#
#
#     a=1
