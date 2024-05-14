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
with open(os.path.join("Slake", "train.json")) as f:
    train_data = json.load(f)
train_dataset = pd.DataFrame(columns = train_data[0].keys())
for sample in train_data:
  train_dataset.loc[len(train_dataset)] = list(sample.values())


#Prepare test dataset
with open(os.path.join("Slake", "validate.json")) as f:
    test_data = json.load(f)
test_dataset = pd.DataFrame(columns = test_data[0].keys())
for sample in test_data:
  test_dataset.loc[len(test_dataset)] = list(sample.values())


#5.6cm focal, predominantly hypodense
array = list(train_dataset['answer'].values)
train_dataset[train_dataset["answer"] == "10-20 minutes"]

#Change column name
train_dataset = train_dataset.rename(columns = {"img_name": "image_id"})
test_dataset = test_dataset.rename(columns = {"img_name": "image_id"})

#Create answer space
train_dup_ans = list(train_dataset["answer"].values)
test_dup_ans = list(test_dataset["answer"].values)

train_answer = list(train_dataset["answer"].unique())
test_answer = list(test_dataset["answer"].unique())

answer_space = list(set(train_answer + test_answer))
answer_space = [str(element) for element in answer_space]
answer_space.sort()

train_dataset = train_dataset.loc[:, ["question", "answer", "image_id"]]
test_dataset = test_dataset.loc[:, ["question", "answer", "image_id"]]
answer_space=['0', '1', '2', '3', '4', '5', '6', 'A Little', 'Abdomen', 'Above the rectum', 'Absorb nutrients, secrete enzymes, digest food', 'Absorb water, excrete body waste', 'Almost the same', 'Around the bladder', 'Atelectasis', 'Atelectasis, Effusion', 'Bacterial infection', 'Barin', 'Biotransformation, detoxification', 'Black', 'Black Hollow', 'Bladder', 'Body', 'Both', 'Both Lung', 'Both Lungs', 'Bottom', 'Brain', 'Brain Edema', 'Brain Edema, Brain Enhancing Tumor', 'Brain Edema, Brain Enhancing Tumor, Brain Non-enhancing Tumor', 'Brain Edema, Brain Non-enhancing Tumor', 'Brain Edema, Brain Tumor', 'Brain Enhancing Tumor, Brain Edema, Brain Non-enhancing Tumor', 'Brain Tumor, Brain Edema', 'Brain embryonic tissue dysplasia, genetic factors, chemical factors', 'Breathe', 'Brian Edema', 'Bronchial obstruction', 'CT', 'Cardiomega', 'Cardiomegal', 'Cardiomegaly', 'Cardiomegaly, Infiltration, Effusion', 'Cardiomegaly, Pneumothorax', 'Center', 'Center, Left Lung', 'Chest', 'Chest injury, lung disease, bullae', 'Chest pain, cough, expectoration', 'Chest pain, dyspnea', 'Chest tightness, arrhythmia', 'Chest tightness, dyspnea, chest pain', 'Chest tightness, fatigue', 'Chest tightness, shortness of breath, difficulty breathing, dry cough', 'Circular', 'Colon', 'Colon, Small Bowel', 'Control heartbeat and breathing', 'Coronal Plane', 'Cough foamy mucus sputum, dyspnea, cough, chest pain', 'Craniocerebral injury, intracranial space-occupying lesions, intracranial inflammation, cerebrovascular lesions, cerebral hypoxia', 'Cutting, chewing, maintaining facial contour and assisting pronunciation', 'Dark Gray', 'Digest food, absorb water, excrete body waste', 'Digest food, secrete enzymes', 'Duodenum', 'Dyspnea, hemoptysis, chest pain', 'Effusion', 'Encephalomalacia, local edema, confusion, increased intracranial pressure', 'Esophagus', 'Eyes', 'Gas delivery', 'Gray', 'Gray ball on the left', 'Grey circle on the left', 'Head', 'Heart', 'Heart and Liver', 'Hyperdense', 'Hypertension, dilated cardiomyopathy', 'Hypodense', "Improve the body's immunity", 'Infiltration', 'Inflammation, bacterial infection, etc', 'Inflammation, malignant tumor, trauma, etc', 'Intestine', 'Irregular', 'Keep healthy', 'Kidney', 'Large Bowel', 'Large Bowel and Stomach', 'Larynx', 'Left', 'Left Femoral Head', 'Left Kidney', 'Left Lobe', 'Left Lung', 'Left Lung, Lower Right', 'Left Lung, Right', 'Left Lung, Upper Right', 'Left Temporal Lobe', 'Left and Top', 'Left and top', 'Light grey', 'Live healthy, enhance physical fitness', 'Liver', 'Liver Cancer', 'Liver Cancer, Kidney Cancer', 'Liver and Heart', 'Liver, Heart, Lung', 'Liver, Heart, Spleen', 'Liver, Heart, Spleen, Lung', 'Liver, Kidney', 'Liver, Left', 'Liver, Stomach', 'Liver, Top', 'Lower Left', 'Lower Left Chest', 'Lower Left Lobe', 'Lower Left Lung', 'Lower Middle', 'Lower Right', 'Lower Right Chest', 'Lower Right Lobe', 'Lower Right Lung', 'Lower right', 'Lower right Lung', 'Lung', 'Lung Cancer', 'Lung, Heart', 'Lung, Liver', 'Lung, Liver, Heart', 'MRI', 'Mandible', 'Mandible, Parotid', 'Mass', 'Mass, Atelectasis', 'Medical therapy, supportive therapy', 'Medical treatment', 'Medical treatment, supportive treatment, surgical treatment', 'Medical treatment, surgical treatment', 'Much', 'Neck', 'No', 'Nodule', 'Not seen', 'Noudle', 'Oval', 'Parotid', 'Participate in hearing, speech and memory', 'Pay attention to dietary hygiene, strengthen physical fitness and avoid brain trauma', 'Pay attention to prevent cold and keep warm, enhance physical fitness', 'Pelvic Cavity', 'Pharmacotherapy, rehabilitation', 'Physical therapy, medication', 'Physical therapy, surgical treatment', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Promote Blood Flow', 'Promote blood flow', 'Pulmonary Bronchus', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Pulmonary Nodule', 'Pulmonary bronchus', 'Pulmonary infection, chronic irritation', 'Pulmonary infection, lung tumor, tuberculosis and other diseases', 'Quit smoking, avoid strenuous exercise', 'Quit smoking, enhance physical fitness', 'Rectum', 'Rectum, Colon, Small Bowel', 'Rectum, Small Bowel', 'Rectum, Small Bowel, Colon', 'Respiratory System', 'Right', 'Right Chest', 'Right Kidney', 'Right Lobe', 'Right Lung', 'Right Lung, Left', 'Right Lung, Lower Left', 'Right Lung, Upper Left', 'Right Lung,Upper Left', 'Right lung', 'Small Bowel', 'Small Bowel, Colon', 'Small Bowel, Colon, Rectum', 'Small Bowel, Duodenum', 'Small Bowel, Rectum', 'Small Bowel,Rectum,Colon', 'Small bowel', 'Spinal Cord', 'Spinal cord', 'Spleen', 'Stomach', 'Stomach, Colon', 'Storage of urine', 'Store feces, excrete feces', 'Store urine', 'Symmetrical to the bone marrow', 'Symmetrical to the bottom spine', 'Symmetrical to the spine', 'T1', 'T2', 'Temporal Lobe', 'Temporal Lobe, Eyes', 'Tooth', 'Top', 'Trachea', 'Transverse  Plane', 'Transverse Plane', 'Treat brain diseases promptly, keep healthy', 'U-shaped', 'Under the trachea', 'Unknown', 'Upper', 'Upper Left', 'Upper Left Lobe', 'Upper Left Lung', 'Upper Right', 'Upper Right Lobe', 'Upper Right Lung', 'Upper left', 'Upper left of spleen', 'Ventilation, pronunciation', 'Visual impairment, vomiting, tinnitus, increased intracranial pressure', 'White', 'X-Ray', 'Yes', 'nan']

from datasets import load_dataset

# Load the training & evaluation dataset present in CSV format
dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join("Slake", "data_train.csv"),
        "test": os.path.join("Slake", "data_eval.csv")
    }
)
none_answer_examples = filter(lambda example: 'answer' in example and example['answer'] is None, dataset['train'])
for example in none_answer_examples:
    print("Found None value in answer field:", example)
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
            context_length = 40
        )
        return {"input_ids": encoded_text}

    def preprocess_images(self, images: List[str]):
        processed_images = [self.preprocessor(Image.open(os.path.join("Slake", "imgs", f"xmlab{img_id}/source.jpg")).convert('RGB')) for img_id in images]
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
                raw_batch_dict['img_id']
                if isinstance(raw_batch_dict, dict) else
                [i['img_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }

"""# Experts"""

# helper functions
from inspect import isfunction
import math
import torch
from torch import nn
import torch.nn.functional as F


from functools import partial
from collections import namedtuple
from typing import Optional, Tuple, Union

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F

from beartype import beartype

from einops import rearrange, repeat, reduce, pack, unpack

from colt5_attention import topk as maybe_differentiable_topk

import torch.distributed as dist
# constants

MIN_EXPERT_CAPACITY = 4

MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
    'total_aux_loss',
    'balance_loss',
    'router_z_loss'
])

# helper functions
class AllGather(nn.Module):
    def __init__(self, *, dim = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x, sizes = None):
        return AllGatherFunction.apply(x, self.dim, sizes)

def split_by_rank(x):
    rank = dist.get_rank()
    out = x[rank]

    if isinstance(x, tuple):
        sizes = tuple(map(lambda t: t.shape[0], x))
    else:
        sizes = (x.shape[1],) * x.shape[0]

    sizes = torch.tensor(sizes, device = out.device, dtype = torch.long)
    return out, sizes
def gather_sizes(t, *, dim):
    size = torch.tensor(t.shape[dim], device = t.device, dtype = torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)
def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))
def has_only_one_value(t):
    return (t == t[0]).all()
def exists(val):
    return val is not None

def default(val, default):
    if exists(val):
        return val

    return default() if callable(default) else default

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cast_tuple(el, len = 1):
    return el if isinstance(el, tuple) else ((el,) * len)

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor related helper functions

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

# rms normalization

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# expert class
# best performing was ff geglu with multiplicative bias (just after gating)

class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias = True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x * self.mult_bias

class Expert(Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)

        self.net = Sequential(
            RMSNorm(dim) if prenorm else None,
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden, mult_bias = mult_bias),
            nn.Linear(dim_hidden, dim)
        )

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        return self.net(x)

class Experts(nn.Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        allow_var_seq_len = False # whether to handle variable sequence length
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        # distributed related settings

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        self.all_gather = AllGather()

        self.allow_var_seq_len = allow_var_seq_len

        # device tracker, since need to manually move experts not in use to CPU in distributed

        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        # declare some variables

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts
        seq_len = shape[-2]

        # for now naively all gather across batch dimension if distributed, optimize later

        world_size = 1
        rank = 0

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            var_seq_len = not has_only_one_value(seq_sizes)

            assert self.allow_var_seq_len or not var_seq_len, 'number of tokens per expert must be the same - if you want the framework to handle it, set `allow_var_seq_len = True` on `Experts`'

            # if variable sequence length, pad

            if var_seq_len:
                max_seq_size = seq_sizes.amax().item()
                x = pad_dim_to(x, max_seq_size, dim = -2)

            # gather and concat across batches, accounting for variable batch sizes

            x, batch_sizes = self.all_gather(x)
            total_batch_size = batch_sizes.sum().item()

            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # the experts in use on the rank

        num_experts_per_rank = num_experts
        expert_slice = slice(0, num_experts)

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim = -1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')

            x = x.split(num_experts_batches_across_ranks, dim = 0)
            x, experts_per_rank_sizes = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []

        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x, requires_grad = self.training)

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs, sizes = experts_per_rank_sizes)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs, _ = split_by_rank(outs)

            # account for padded sequence length
            outs = outs[..., :seq_len, :]

        assert outs.shape == shape
        return outs

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class TopNGating(Module):

    @beartype
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        top_n = 2,
        threshold_train: Union[float, Tuple[float, ...]] = 0.2,
        threshold_eval: Union[float, Tuple[float, ...]] = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.to_gates = nn.Linear(dim, num_gates, bias = False)

        self.differentiable_topk = differentiable_topk

        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable = not differentiable_topk,
            fused = differentiable_topk_fused # use triton fused coordinate descent if possible by default
        )

        assert top_n >= 2, 'must be 2 or more experts'
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval        

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        """

        *_, b, group_size, dim, dtype, top_n, num_gates, eps = *x.shape, x.dtype, self.top_n, self.num_gates, self.eps

        # threshold, capacity depending on training or eval

        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        gate_logits = self.to_gates(x)

        maybe_noised_gate_logits = gate_logits

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim = -1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k = top_n)

        gate_indices = topk_return.indices

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        # move the top-n dimension to be first

        gates = rearrange(gates, '... k -> k ...')
        gate_indices = rearrange(gate_indices, '... k -> k ...')

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0] # needed for balancing loss

        # normalize top-n gate scores

        denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min = eps)
        gates = gates / denom

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        probs = torch.zeros_like(gates).uniform_(0., 1.)

        threshold = rearrange(threshold, 'k -> k 1 1')
        should_route = probs < (gates / threshold.clamp(min = eps))

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True

        mask *= rearrange(should_route.float(), '... -> ... 1')

        mask_cumsum = cumsum_exclusive(mask, dim = -2) # along sequence dimension

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float()

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = reduce(mask[n], '... n e -> ... 1 e', 'sum')

            # (batch, sequence)
            position_in_expert = reduce(position_in_expert, '... n e -> ... n', 'sum')
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, '... n e -> ... n', 'sum')

        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        N = None

        gates = gates[..., N, N]
        mask_flat = mask_flat[..., N, N]
        one_hot_gate_indices = one_hot_gate_indices[..., N]
        safe_one_hot_gates = safe_one_hot(positions.long(), expert_capacity)[..., N, :]

        combine_tensor = reduce(
            gates
            * mask_flat
            * one_hot_gate_indices
            * safe_one_hot_gates
        , 'k ... -> ...', 'sum')

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
            density_1_proxy = reduce(raw_gates, '... n e -> ... e', 'mean') # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim = -1)
            router_z_loss = torch.square(router_z_loss)            
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss

# plain mixture of experts

class MoE(Module):

    @beartype
    def __init__(self,
        dim,
        num_experts = 16,
        expert_hidden_mult = 4,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        gating_top_n = 2,
        balance_loss_coef = 1e-2,
        router_z_loss_coef = 1e-3,
        experts: Optional[Module] = None,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        is_distributed = None,
        allow_var_seq_len = False
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        self.gate = TopNGating(
            dim,
            top_n = gating_top_n,
            num_gates = num_experts,
            straight_through_dispatch_tensor = straight_through_dispatch_tensor,
            differentiable_topk = differentiable_topk,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        experts = default(experts, lambda: [Expert(dim = dim, hidden_mult = expert_hidden_mult) for _ in range(num_experts)])

        self.experts = Experts(
            experts,
            is_distributed = is_distributed,
            allow_var_seq_len = allow_var_seq_len
        )

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(x, noise_gates = noise_gates, noise_mult = noise_mult)

        # dispatch

        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)

        # feed the expert inputs through the experts.

        expert_outputs = self.experts(expert_inputs)

        # combine

        output = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)

        # losses

        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses

        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        return MixtureOfExpertsReturn(output, total_aux_loss, balance_loss, router_z_loss)

# sparse moe block
# in particular, they found that adding a feedforward before or after greatly stabilized the training and improved results

class SparseMoEBlock(Module):

    @beartype
    def __init__(
        self,
        moe: MoE,
        *,
        add_ff_before = False,
        add_ff_after = True
    ):
        super().__init__()
        dim = moe.dim

        self.moe = moe
        self.moe_prenorm = RMSNorm(dim)

        self.ff_before = Expert(dim, prenorm = True) if add_ff_before else None
        self.ff_after = Expert(dim, prenorm = True) if add_ff_after else None

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):

        # feedforward before

        if exists(self.ff_before):
            x = self.ff_before(x) + x

        # mixture of experts layer

        residual = x

        moe_out, total_aux_loss, balance_loss, router_z_loss = self.moe(self.moe_prenorm(x), noise_gates = noise_gates, noise_mult = noise_mult)

        x = moe_out + residual

        # feedforward after

        if exists(self.ff_after):
            x = self.ff_after(x) + x

        return MixtureOfExpertsReturn(x, total_aux_loss, balance_loss, router_z_loss)
"""# Ensemble models"""

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4* n_embed),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.3))

    def forward(self, x):
        return self.net(x)

# class TextAdapter(nn.Module):
#     def __init__(self, n_embed):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(4 * n_embed, n_embed),
#             nn.Dropout(0.3))

#     def forward(self, x):
#         return self.net(x)


# class SelfAttentionBlock(nn.Module):
#     def __init__(self, embed_size, num_heads):
#         super(SelfAttentionBlock, self).__init__()
#         self.embed_size = embed_size
#         self.num_heads = num_heads

#         self.query_linear = nn.Linear(embed_size, embed_size)
#         self.key_linear = nn.Linear(embed_size, embed_size)
#         self.value_linear = nn.Linear(embed_size, embed_size)


#         self.first_linear = nn.Linear(embed_size, embed_size)

#         self.multiheadAttention = nn.MultiheadAttention(embed_size, num_heads)
#         self.first_norm = nn.LayerNorm(embed_size)

#         self.feed_forward = FeedForward(embed_size)
#         self.final_norm = nn.LayerNorm(embed_size)

#     def forward(self, x):

#         Q = self.query_linear(x)
#         K = self.key_linear(x)
#         V = self.value_linear(x)


#         Q = Q.permute(1, 0, 2)
#         K = K.permute(1, 0, 2)
#         V = V.permute(1, 0, 2)


#         attn_output, _ = self.multiheadAttention(Q, K, V)

#         # Transpose back to [batch_size, seq_len, embed_size] and apply linear transformation
#         attn_output = self.first_linear(attn_output.permute(1, 0, 2))
#         attn_output = self.first_norm(attn_output + x)

#         ff_output = self.feed_forward(attn_output)
#         final = self.final_norm(attn_output + ff_output)

#         return final

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
        self.project = MoE(dim=embed_size, num_experts=num_experts, balance_loss_coef = aux_loss_coef,
                           gating_top_n = 3,
                           capacity_factor_train = 3,
                           threshold_train = 0.05,
                           threshold_eval = 0.05,
                           experts = torch.nn.ModuleList(
                                     [FeedForward(embed_size) for _ in range(num_experts)]
                                     ))

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

        attn_out, aux_l,_,_ = self.project(attn_output)

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

# class Multi_project(nn.Module):
#   def __init__(self, input, output):
#     super().__init__()
#     self.project = nn.Linear(input, output)

#   def forward(self, x):
#     return self.project(x)

#model = MultimodalVQAModel('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

#num_trainable_params = count_trainable_parameters(model)

"""# Model architecture"""

class MultimodalVQAModel(nn.Module):
    def __init__(self,  pretrained_clip_name, num_labels=len(answer_space), intermediate_dim=512, num_heads=12, num_experts=8, aux_loss_coef = 1e-2):
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

        self.image_adapter = MoE(dim=self.embed_dim, num_experts=num_experts, balance_loss_coef = aux_loss_coef,
                                 gating_top_n = 3,
                                 capacity_factor_train = 3,
                                 threshold_train = 0.05,
                                 threshold_eval = 0.05,
                                 experts = torch.nn.ModuleList(
                                     [FeedForward(self.embed_dim) for _ in range(num_experts)]
                                     ))
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
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim, 1)
        )

        self.mlp_img = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
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
        encoded_image, moe_loss,_,_ = self.image_adapter(raw_image)
        encoded_image = self.moe_norm(encoded_image + raw_image)
        # image_features = self.i_norm(self.img_proj(encoded_image) + encoded_image)  #(batch, length, dim)

        query_refine, image_refine, l1 = self.first_modular_block(encoded_text, encoded_image)
        final_query, final_image, l2 = self.second_modular_block(query_refine, image_refine)
        final_query, final_image, l3 = self.third_modular_block(final_query, final_image)
        # final_query, final_image, l4 = self.fourth_modular_block(final_query, final_image)

       #final_query = (batch, query_length, dim), final_image = (batch, image_length, dim)

        query_weights = F.softmax(self.mlp_text(final_query), dim = 1)
        img_weights = F.softmax(self.mlp_img(final_image), dim = 1)

        query = torch.sum(query_weights * final_query, dim = 1) #(batch, dim)
        image = torch.sum(img_weights * final_image, dim = 1)   #(batch, dim)

        q = self.final_proj_text(query)
        i = self.final_proj_img(image)

        modular_loss = l1 + l2 + l3 + moe_loss

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
    num_train_epochs=40,
    #fp16=True,
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
