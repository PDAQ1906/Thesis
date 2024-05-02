import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

from sklearn.metrics import accuracy_score, f1_score

# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_caching_enabled(True)
logging.set_verbosity_error()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import pandas as pd
import json
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#Prepare train dataset
with open(os.path.join("data_RAD", "trainset.json")) as f:
    train_data = json.load(f)
train_dataset = pd.DataFrame(columns = train_data[0].keys())
for sample in train_data:
  train_dataset.loc[len(train_dataset)] = list(sample.values())


#Prepare test dataset
with open(os.path.join("data_RAD", "testset.json")) as f:
    test_data = json.load(f)
test_dataset = pd.DataFrame(columns = test_data[0].keys())
for sample in test_data:
  test_dataset.loc[len(test_dataset)] = list(sample.values())

#Fix error
train_dataset = train_dataset.replace(" superficial to the patient's skin", "superficial to the patient's skin")

#5.6cm focal, predominantly hypodense
array = list(train_dataset['answer'].values)
train_dataset[train_dataset["answer"] == "10-20 minutes"]

#Change column name
train_dataset = train_dataset.rename(columns = {"image_name": "image_id"})
test_dataset = test_dataset.rename(columns = {"image_name": "image_id"})

#Create answer space
train_dup_ans = list(train_dataset["answer"].values)
test_dup_ans = list(test_dataset["answer"].values)

train_answer = list(train_dataset["answer"].unique())
test_answer = list(test_dataset["answer"].unique())

answer_space = list(set(train_answer + test_answer))
answer_space = [str(element) for element in answer_space]
answer_space.sort()

#Save files

# with open(os.path.join("data_RAD", "answer_space.txt"), "w") as f:
#     f.writelines("\n".join(answer_space))

train_dataset = train_dataset.loc[:, ["question", "answer", "image_id"]]
test_dataset = test_dataset.loc[:, ["question", "answer", "image_id"]]

# train_dataset.to_csv(os.path.join("data_RAD", "data_train.csv"), index=None)
# test_dataset.to_csv(os.path.join("data_RAD", "data_eval.csv"), index=None)

from datasets import load_dataset

# Load the training & evaluation dataset present in CSV format
dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join("data_RAD", "data_train.csv"),
        "test": os.path.join("data_RAD", "data_eval.csv")
    }
)

#Turn answers to labels
dataset = dataset.map(
    lambda examples: {
        'label': [
            answer_space.index(ans)
            for ans in examples['answer']
        ]
    },
    batched=True
)

"""# Tokenizer Initialization"""

from dataclasses import dataclass
from typing import List
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

@dataclass
class MultimodalCollator:
    tokenizer: get_tokenizer
    preprocessor: create_model_from_pretrained

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            texts=texts,
            context_length = 30
        )
        return {"input_ids": encoded_text}

    def preprocess_images(self, images: List[str]):
        processed_images = [self.preprocessor(Image.open(os.path.join("data_RAD", "images", image_id)).convert('RGB')) for image_id in images]
        # Assuming the preprocessor returns a processed image tensor, stack them if necessary
        return {"pixel_values": torch.stack(processed_images)}

    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from inspect import isfunction
import math



class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4* n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.4))

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, num_experts, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        # self.output_size = output_size
        self.input_size = input_size
        # self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([FeedForward(self.input_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss

# helper functions
from inspect import isfunction
import math
import torch
from torch import nn
import torch.nn.functional as F
from mixture_of_experts import MoE

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)
        print(hidden_dim, num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

"""# Attention networks"""

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4* n_embed),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.3))

    def forward(self, x):
        return self.net(x)

class TextAdapter(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.3))

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)


        self.first_linear = nn.Linear(embed_size, embed_size)

        self.multiheadAttention = nn.MultiheadAttention(embed_size, num_heads)
        self.first_norm = nn.LayerNorm(embed_size)

        self.feed_forward = FeedForward(embed_size)
        self.final_norm = nn.LayerNorm(embed_size)

    def forward(self, x):

        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)


        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)


        attn_output, _ = self.multiheadAttention(Q, K, V)

        # Transpose back to [batch_size, seq_len, embed_size] and apply linear transformation
        attn_output = self.first_linear(attn_output.permute(1, 0, 2))
        attn_output = self.first_norm(attn_output + x)

        ff_output = self.feed_forward(attn_output)
        final = self.final_norm(attn_output + ff_output)

        return final

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_size, num_heads, num_experts, aux_loss_coef = 1e-2):
        super(CrossAttentionBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads


        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)


        self.multiheadAttention = nn.MultiheadAttention(embed_size, num_heads)
        self.adapt = nn.Linear(embed_size, embed_size)
        self.project = MoE(dim=embed_size, num_experts=num_experts, loss_coef = aux_loss_coef, experts = FeedForward(embed_size))

        self.first_norm = nn.LayerNorm(embed_size)
        self.norm = nn.LayerNorm(embed_size)


        # self.feed_forward = MoE(dim=embed_size, num_experts=num_experts, loss_coef = aux_loss_coef)
        # self.final_norm = nn.LayerNorm(embed_size)

    def forward(self, query, key_value):

        Q = self.query_linear(query).permute(1, 0, 2)     #(batch, query_length, dim)
        K = self.key_linear(key_value).permute(1, 0, 2)   #(batch, key_length, dim)
        V = self.value_linear(key_value).permute(1, 0, 2) #(batch, key_length, dim)


        attn_output, _ = self.multiheadAttention(Q, K, V) #(batch, dim, length)
        attn_output = self.first_norm(self.adapt(attn_output.permute(1, 0, 2)) + query)

        attn_out, aux_l = self.project(attn_output)

        # attn_output = attn_output.permute(1, 0, 2)
        attn_out = self.norm(attn_out + attn_output)


        # ff_output, aux_l2 = self.feed_forward(attn_out)
        # output = self.final_norm(ff_output + attn_out)

        return attn_out, aux_l


class ModularAttentionBlock(nn.Module):
  def __init__(self, embed_size, num_heads, num_experts, aux_loss_coef=1e-2):
        super(ModularAttentionBlock, self).__init__()

        # self.question_attn = SelfAttentionBlock(embed_size, num_heads)
        self.image_attn = CrossAttentionBlock(embed_size, num_heads, num_experts, aux_loss_coef)

        self.question_attn = CrossAttentionBlock(embed_size, num_heads, num_experts, aux_loss_coef)

  def forward(self, question, image):
    query, aux_loss1 = self.question_attn(question, image) #(batch, query_length, dim)
    value, aux_loss2 = self.image_attn(image, question)    #(batch, image_length, dim)

    return query, value, aux_loss1 + aux_loss2

"""# Model architecture"""

class MultimodalVQAModel(nn.Module):
    def __init__(self,  pretrained_clip_name, num_labels=len(answer_space), intermediate_dim=512, num_heads=12, num_experts=4, aux_loss_coef = 0.15):
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_clip_name = pretrained_clip_name
        self.intermediate_dim = intermediate_dim
        # self.proj_dim = proj_dim

	# Pretrained transformers for text & image featurization
        model, _ = create_model_from_pretrained(self.pretrained_clip_name)

        self.embed = model.text.transformer.embeddings
        self.text_encoder = model.text.transformer.encoder
        self.image_encoder = model.visual.trunk #VisionTransformer model

        for param in self.embed.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        #Train text_encoder 2 last layers
        for block_num in [-3, -2, -1]:
            for param in self.text_encoder.layer[block_num].parameters():
                param.requires_grad = True

        #Train image_encoder 2 last layers
        for block_num in [-3, -2, -1]:
            for param in self.image_encoder.blocks[block_num].parameters():
                param.requires_grad = True
        #Train image_encoder norm layer
        for param in self.image_encoder.norm.parameters():
            param.requires_grad = True

        self.embed_dim = self.image_encoder.embed_dim

        self.image_encoder.blocks[-1].mlp = nn.Identity()

        self.image_adapter = MoE(dim=self.embed_dim, num_experts=num_experts*2,
                                loss_coef = aux_loss_coef, experts = FeedForward(self.embed_dim))
        self.moe_norm = nn.LayerNorm(self.embed_dim)
        # self.text_encoder.layer[-1].output = MoE(dim=self.embed_dim, num_experts=num_experts,
        #                                         loss_coef = aux_loss_coef, experts = TextAdapter(self.embed_dim))


  #Projection
        # self.img_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.img_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.text_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # self.i_norm = nn.LayerNorm(self.embed_dim)
        # self.t_norm = nn.LayerNorm(self.embed_dim)

  # Cross-attention layer
        self.first_modular_block = ModularAttentionBlock(self.embed_dim, num_heads, num_experts, aux_loss_coef)
        self.second_modular_block = ModularAttentionBlock(self.embed_dim, num_heads, num_experts, aux_loss_coef)
        self.third_modular_block = ModularAttentionBlock(self.embed_dim, num_heads, num_experts, aux_loss_coef)
        # self.fourth_modular_block = ModularAttentionBlock(self.embed_dim, num_heads, num_experts, aux_loss_coef)


        self.mlp_text = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, 1)
        )

        self.mlp_img = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, 1)
        )

        self.final_proj_text = nn.Linear(self.embed_dim, intermediate_dim)
        self.final_proj_img = nn.Linear(self.embed_dim, intermediate_dim)


	# Fusion layer for cross-modal interaction
        # self.fusion = nn.Sequential(
        #     nn.Linear(self.embed_dim*2, intermediate_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.4),
        # )

	# Fully-connected classifier
        self.norm = nn.LayerNorm(intermediate_dim)
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)

        self.criterion = nn.CrossEntropyLoss()


    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.LongTensor,
            labels: Optional[torch.LongTensor] = None):

       #Question representation
        encoded_text = self.embed(input_ids)
        encoded_text = self.text_encoder(encoded_text).last_hidden_state      #(batch, seq_length, dim)
        # text_features = self.t_norm(self.text_proj(encoded_text) + encoded_text)

       #Visual representation
        raw_image = self.image_encoder.forward_features(pixel_values)          #(batch, length, dim)
        encoded_image, moe_loss = self.image_adapter(raw_image)
        encoded_image = self.moe_norm(encoded_image + raw_image)
        # image_features = self.i_norm(self.img_proj(encoded_image) + encoded_image)  #(batch, length, dim)

        query_refine, image_refine, l1 = self.first_modular_block(encoded_text, encoded_image)
        final_query, final_image, l2 = self.second_modular_block(query_refine, image_refine)
        final_query, final_image, l3 = self.third_modular_block(final_query, final_image)
        # final_query, final_image, l4 = self.fourth_modular_block(final_query, final_image)

        modular_loss = l1 + l2 + l3 + moe_loss
       #final_query = (batch, query_length, dim), final_image = (batch, image_length, dim)

        query_weights = F.softmax(self.mlp_text(final_query), dim = 1)
        img_weights = F.softmax(self.mlp_img(final_image), dim = 1)

        query = torch.sum(query_weights * final_query, dim = 1) #(batch, dim)
        image = torch.sum(img_weights * final_image, dim = 1)   #(batch, dim)

        q = self.final_proj_text(query)
        i = self.final_proj_img(image)

        # fused_output = self.fusion(torch.cat([
        #     final_query[:,0], final_image[:,0]
        # ],
        #     dim = 1)
        # )

        fused_output = self.norm(q + i)

        logits = self.classifier(fused_output)

        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss + modular_loss

        return out

"""## Evaluation metrics"""

def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a)
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score

# Wrapper around the wup_measure(...) function to process batch inputs
def batch_wup_measure(labels, preds):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)

# Function to compute all relevant performance metrics, to be passed into the trainer
def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    preds = logits.argmax(axis=-1)
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }

def createMultimodalVQACollatorAndModel(clip_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'):
    # Initialize the correct text tokenizer and image feature extractor, and use them to create the collator

    model, preprocessor = create_model_from_pretrained(clip_name)
    tokenizer = get_tokenizer(clip_name)

    #Vision-Language Tokenizer
    multimodal_collator = MultimodalCollator(tokenizer=tokenizer, preprocessor=preprocessor)

    # Initialize the multimodal model with the appropriate weights from pretrained models
    multimodal_model = MultimodalVQAModel(clip_name).to(device)


    #Return Vision-Language Tokenizer and Vision-Language model
    return multimodal_collator, multimodal_model

import os
from transformers import TrainingArguments

# Determine the number of available CPU cores
num_cpu_cores = os.cpu_count()
num_cpu_cores

from transformers import TrainingArguments, AutoTokenizer, AutoFeatureExtractor


args = TrainingArguments(
    output_dir="checkpoint",
    seed=12345,
    evaluation_strategy="steps",
    eval_steps=96,                # Every half-epoch, for more frequent monitoring
    logging_strategy="steps",
    logging_steps=96,             # Log every half-epoch
    save_strategy="steps",
    save_steps=192,               # Save checkpoints every full epoch
    save_total_limit=2,           # Keep only the last 2 checkpoints to manage disk space
    metric_for_best_model='wups',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    num_train_epochs=50,
    fp16=True,
    dataloader_num_workers=2,
    load_best_model_at_end=True,
)

# Initialize the actual collator and multimodal model
collator, model = createMultimodalVQACollatorAndModel()

# Initialize the trainer with the dataset, collator, model, hyperparameters and evaluation metrics
multi_args = deepcopy(args)
multi_args.output_dir = os.path.join("checkpoint", "BiomedCLIP")
multi_trainer = Trainer(
    model,
    multi_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=collator,
    compute_metrics=compute_metrics
)

"""# Training"""

# Start the training loop
train_multi_metrics = multi_trainer.train()

# Run the model on the evaluation set to obtain final metrics
eval_multi_metrics = multi_trainer.evaluate()

torch.cuda.memory_summary(device=None, abbreviated=False)

torch.save(model.state_dict(), os.path.join(multi_args.output_dir, "pytorch_model.bin"))
