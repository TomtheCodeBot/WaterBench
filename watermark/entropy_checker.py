# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
from math import sqrt
from itertools import chain, tee
from functools import lru_cache
from nltk import pos_tag, word_tokenize, RegexpParser

import scipy.stats
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor
import torch.nn.functional as F


class POSEntropyChecker( LogitsProcessor):


    def __init__(self, tokenizer, store_spike_ents: bool = False, **kwargs):
        super().__init__( **kwargs)
        self.entropy_per_tag = {}
        self.tokenizer = tokenizer
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""
        top_score, next_token = torch.sort(scores, dim=1,descending=True)
        next_token = next_token[:,0]
        output_score = torch.zeros(scores.shape)
        new_cuurrent_char = {}
        for b_idx in range(input_ids.shape[0]):
            sliced_input = input_ids[b_idx][self.prompt_slice:]
            tokens = sliced_input
            
            text = self.tokenizer.decode(tokens)
            #print(pos_tag(word_tokenize(text)))
            next_output = self.tokenizer.convert_ids_to_tokens(next_token[b_idx].item())
            #print(next_output)
            curr_pos_tag = pos_tag(word_tokenize(text))
            if len(next_output)==1  or next_output[0]!="▁" or len(tokens)==0 or len(curr_pos_tag)==0:
                continue
            if len(tokens)!=0:
                curr_word ,current_tag = curr_pos_tag[-1]
            flag = 0
            # Convert logits to probabilities using softmax
            probabilities = F.softmax(scores[b_idx], dim=-1)
                
            # Calculate entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1).item()
            flag=0
            if not current_tag in self.entropy_per_tag.keys():
                self.entropy_per_tag[current_tag] = [entropy]
            else:
                self.entropy_per_tag[current_tag].append(entropy)
        return scores
