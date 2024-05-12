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

train_dataset = train_dataset.loc[:, ["question", "answer", "image_id"]]
test_dataset = test_dataset.loc[:, ["question", "answer", "image_id"]]

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
            context_length = 40
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

"""# Experts"""

# helper functions
from inspect import isfunction
import math
import torch
from torch import nn
import torch.nn.functional as F
from st_moe_pytorch import MoE

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

model = MultimodalVQAModel('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

num_trainable_params = count_trainable_parameters(model)

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
