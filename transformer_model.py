import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, maxlen = 5000):
        super(PositionalEncoding, self).__init__()

        # A tensor consists of all the possible positions (index) e.g 0, 1, 2, ... max length of input
        # Shape (pos) --> [max len, 1]
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_encoding = torch.zeros((maxlen, d_model))

        sin_den = 10000 ** (torch.arange(0, d_model, 2)/d_model) # sin for even item of position's dimension
        cos_den = 10000 ** (torch.arange(1, d_model, 2)/d_model) # cos for odd

        pos_encoding[:, 0::2] = torch.sin(pos / sin_den)
        pos_encoding[:, 1::2] = torch.cos(pos / cos_den)

        # Shape (pos_embedding) --> [max len, d_model]
        # Adding one more dimension in-between
        pos_encoding = pos_encoding.unsqueeze(-2)
        # Shape (pos_embedding) --> [max len, 1, d_model]

        self.dropout = nn.Dropout(dropout)

        # We want pos_encoding be saved and restored in the `state_dict`, but not trained by the optimizer
        # hence registering it!
        # Source & credits: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/2
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding):
        # shape (token_embedding) --> [sentence len, batch size, d_model]

        # Concatenating embeddings with positional encodings
        # Note: As we made positional encoding with the size max length of sentence in our dataset
        #       hence here we are picking till the sentence length in a batch
        #       Another thing to notice is in the Transformer's paper they used FIXED positional encoding,
        #       there are methods where we can also learn them
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TransformerClassifier(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 d_model,
                 dropout,
                 n_head,
                 dim_feedforward,
                 n_layers
                ):
        super().__init__()

        # self.src_inp_emb = InputEmbedding(src_vocab_size, d_model)
        # self.trg_inp_emb = InputEmbedding(trg_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Only using Encoder of Transformer model
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.d_model = d_model
        self.decoder = nn.Linear(d_model, trg_vocab_size)

    def forward(self, x):
        x_emb = self.positional_encoding(x)
        # Shape (output) -> (Sequence length, batch size, d_model)
        output = self.transformer_encoder(x_emb)
        # We want our output to be in the shape of (batch size, d_model) so that
        # we can use it with CrossEntropyLoss hence averaging using first (Sequence length) dimension
        # Shape (mean) -> (batch size, d_model)
        # Shape (decoder) -> (batch size, d_model)
        # For binary classification, if for multi-class, should change it to softmax.
        return torch.sigmoid(self.decoder(output.mean(0)))
