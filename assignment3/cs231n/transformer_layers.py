import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""


class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """

    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # for i in range(max_len):
        #     for j in range(embed_dim):
        #         if i % 2 == 0:
        #             pe[0, i, j] = math.sin(i * 10000 ** (-j / embed_dim))
        #         else:
        #             pe[0, i, j] = math.cos(i * 10000 ** (-(j - 1) / embed_dim))

        # pe[]
        for i in range(max_len):
            for j in range(embed_dim):
                if j % 2 == 0:
                    pe[0, i, j] += math.sin(i * 10000 ** (-j / embed_dim))
                else:
                    pe[0, i, j] += math.cos(i * 10000 ** (-(j - 1) / embed_dim))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for name, buff in self.named_buffers():
            # print(name)
            # print(buff.shape)
            # print(x.shape)
            output = x + buff[:, :S, :D]

        output = self.dropout(output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        ############################################################################
        # TODO: Initialize any remaining layers and parameters to perform the      #
        # attention operation as defined in Transformer_Captioning.ipynb. We will  #
        # also apply dropout just after the softmax step. For reference, our       #
        # solution is less than 5 lines.                                           #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.alignment = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.Softmax(dim=1)
        self.dropout_attn = nn.Dropout(p=dropout)
        self.context_vectors = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads

        # self.alignment = self.query * self.key / math.sqrt(embed_dim)
        # self.attention = torch.softmax(self.alignment, dim=1)
        # self.dropout_attn = torch.dropout(self.attention, dropout, train=True)
        # self.context_vectors = self.dropout_attn @ self.value

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (T, S) where mask[i,j] == 0 indicates token
          i in the target should not be influenced by token j in the source.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, D = query.shape
        N, T, D = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, T, D))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Case:
        # self.alignment = self.query * self.key / math.sqrt(embed_dim)
        # self.attention = torch.softmax(self.alignment, dim=1)
        # self.dropout_attn = torch.dropout(self.attention, dropout, train=True)
        # self.context_vectors = self.dropout_attn @ self.value

        # Inputs:
        #  - query: shape (N, S, E)
        #  - key: shape (N, T, E)
        #  - value: shape (N, T, E)
        #  - attn_mask: shape (T, S)
        # Returns:
        #  - output: shape (N, S, E)

        # # First, do the linear layers to get key/query/value based on input data:
        # key = self.key(key)
        # query = self.query(query)
        # value = self.value(value)
        # if attn_mask is not None:
        #     pass  # TODO: something with self.proj?

        # FIRST:
        # split into N, H, T, E/H
        for head in range(self.num_heads):
            # head 1
            pass
            # head 2
            pass

        # Then, the logic:
        alignment = self.alignment(query * key / math.sqrt(D))
        attention = self.attention(alignment)
        dropout_attn = self.dropout_attn(attention)
        # print(dropout_attn.T.shape)
        # print(value.shape)

        # torch.masked_fill()
        # output = self.context_vectors(dropout_attn.transpose(-1, -2) @ value)
        # PyTorch supports "None"-style indexing for new axes:
        #   https://sparrow.dev/adding-a-dimension-to-a-tensor-in-pytorch/
        output = self.context_vectors(
            torch.sum(dropout_attn[:, :, :, None] @ value.unsqueeze(dim=2), dim=3)
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


# DEBUG:
if __name__ == "__main__":
    print("in main")
    import numpy as np

    def rel_error(x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    torch.manual_seed(231)

    # Choose dimensions such that they are all unique for easier debugging:
    # Specifically, the following values correspond to N=1, H=2, T=3, E//H=4, and E=8.
    batch_size = 1
    sequence_length = 3
    embed_dim = 8
    attn = MultiHeadAttention(embed_dim, num_heads=2)

    # Self-attention.
    data = torch.randn(batch_size, sequence_length, embed_dim)
    self_attn_output = attn(query=data, key=data, value=data)

    # Masked self-attention.
    mask = torch.randn(sequence_length, sequence_length) < 0.5
    masked_self_attn_output = attn(query=data, key=data, value=data, attn_mask=mask)

    # Attention using two inputs.
    other_data = torch.randn(batch_size, sequence_length, embed_dim)
    attn_output = attn(query=data, key=other_data, value=other_data)

    expected_self_attn_output = np.asarray(
        [
            [
                [-0.2494, 0.1396, 0.4323, -0.2411, -0.1547, 0.2329, -0.1936, -0.1444],
                [-0.1997, 0.1746, 0.7377, -0.3549, -0.2657, 0.2693, -0.2541, -0.2476],
                [-0.0625, 0.1503, 0.7572, -0.3974, -0.1681, 0.2168, -0.2478, -0.3038],
            ]
        ]
    )
    print(
        "self_attn_output error: ",
        rel_error(expected_self_attn_output, self_attn_output.detach().numpy()),
    )

    expected_masked_self_attn_output = np.asarray(
        [
            [
                [-0.1347, 0.1934, 0.8628, -0.4903, -0.2614, 0.2798, -0.2586, -0.3019],
                [-0.1013, 0.3111, 0.5783, -0.3248, -0.3842, 0.1482, -0.3628, -0.1496],
                [-0.2071, 0.1669, 0.7097, -0.3152, -0.3136, 0.2520, -0.2774, -0.2208],
            ]
        ]
    )
    print(
        "masked_self_attn_output error: ",
        rel_error(
            expected_masked_self_attn_output, masked_self_attn_output.detach().numpy()
        ),
    )

    expected_attn_output = np.asarray(
        [
            [
                [-0.1980, 0.4083, 0.1968, -0.3477, 0.0321, 0.4258, -0.8972, -0.2744],
                [-0.1603, 0.4155, 0.2295, -0.3485, -0.0341, 0.3929, -0.8248, -0.2767],
                [-0.0908, 0.4113, 0.3017, -0.3539, -0.1020, 0.3784, -0.7189, -0.2912],
            ]
        ]
    )

    # print('self_attn_output error: ', rel_error(expected_self_attn_output, self_attn_output.detach().numpy()))
    # print('masked_self_attn_output error: ', rel_error(expected_masked_self_attn_output, masked_self_attn_output.detach().numpy()))
    print(
        "attn_output error: ",
        rel_error(expected_attn_output, attn_output.detach().numpy()),
    )

    print()
    print(expected_self_attn_output.shape)
    print(self_attn_output.shape)
