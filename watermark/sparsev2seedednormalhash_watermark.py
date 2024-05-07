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
from math import sqrt,ceil

import scipy.stats
import bitarray
from nltk import pos_tag, word_tokenize, RegexpParser
from string import punctuation
from Levenshtein import distance


import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor
from typing import List

from nltk.util import ngrams

from watermark.normalizers import normalization_strategy_lookup
from .additional_utils.alternative_prf_schemes import prf_lookup, seeding_scheme_lookup

def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += int(bit)*(2**i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]
class SparseV2RandomNormalHashLogitsProcessor:
    def __init__(self,tokenizer,secret_watermark,prompt_slice,bitnum=3,allowed_pos_tag=["V"],granularity = "word",index = 0,shifted = 0,random_bit_string=False,seeding_scheme="lefthash"):
        
        self.tokenizer = tokenizer
        self.input_index = index
        if random_bit_string:
            self.bit_stream = list("10010010011000011101")
        else:
            self.secret_watermark = secret_watermark
            self.ba = bitarray.bitarray()
            self.ba.frombytes(self.secret_watermark.encode('utf-8'))
            self.bit_stream = self.ba.tolist()
        self.num_bit = bitnum
        #self.bit_stream.extend(["0"for _ in range(len(self.bit_stream)%self.num_bit)])

        self.bit_stream.extend(["0"for _ in range(self.num_bit-len(self.bit_stream)%self.num_bit)])
        self.cuurrent_char = None
        self.prompt_slice = prompt_slice
        self.allowed_pos_tag = allowed_pos_tag
        self.prev_encode_action = False
        self.hard_encode = True
        self.last_input = []
        self.new_line = ["<0x0A>"]
        self.prev_check = False
        self.device = "cuda"
        self.rng = torch.Generator(device=self.device)
        self._initialize_seeding_scheme(seeding_scheme)
    def init_table(self):
        self.table = []
        for i in range(len(self.tokenizer.get_vocab().keys())):
            if self.tokenizer.convert_ids_to_tokens(i)[0] != "▁" or len(self.tokenizer.convert_ids_to_tokens(i))==1 or not self.tokenizer.convert_ids_to_tokens(i)[1].isalpha()  :
                continue
            else:
                self.table.append(i)
        
        self.table_size = len(self.table)
        self.table = torch.tensor(self.table).to(self.device)
    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG.")

        prf_key = prf_lookup[self.prf_type](input_ids[-self.context_width :], salt_key=self.hash_key)
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)

        greenlist_size = int(self.table_size)
        vocab_permutation = torch.randperm(self.table_size, device=input_ids.device, generator=self.rng)
        
        return self.table[vocab_permutation]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_score, next_token = torch.sort(scores, dim=1,descending=True)
        next_token = next_token[:,0]
        output_score = torch.zeros(scores.shape)
        new_cuurrent_char = {}
        for b_idx in range(input_ids.shape[0]):
            sliced_input = input_ids[b_idx][self.prompt_slice:]
            tokens = sliced_input
            if self.cuurrent_char is None:
                self.cuurrent_char={}
                self.cuurrent_char[self.tokenizer.decode(tokens[:-2])] = 0
            text = self.tokenizer.decode(tokens)
            #print(pos_tag(word_tokenize(text)))
            
            next_output = self.tokenizer.convert_ids_to_tokens(next_token[b_idx].item())
            #print(next_output)
            if self.prev_encode_action:
                
                secret_character = self.bit_stream[(self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]
                inner_tokens = word_tokenize(text)
                if inner_tokens[-1][0].lower() in punctuation:

                    prev_word , check_tag = pos_tag(inner_tokens)[-2]
                elif self.prev_check and self.tokenizer.convert_ids_to_tokens(tokens[-1].item())[0]=="▁":
                    
                    prev_word , check_tag = pos_tag(inner_tokens)[-2]
                else:
                    prev_word , check_tag = pos_tag(inner_tokens)[-1]
                if not (prev_word[0].isnumeric() or prev_word[0] in punctuation+"”"+'“'):
                    encoded_bit = 2**self.num_bit-1
                    for i in range (2**self.num_bit):
                        if tokens[-1] in self.current_ids[i]:
                            encoded_bit = i
                            break

                    if self.hard_encode:
                        if encoded_bit == bits2int(secret_character):
                            self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                            self.last_input.append(prev_word.lower())
                        else:
                            print(prev_word,encoded_bit,bits2int(secret_character))
                            self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                            #raise Exception
                    else:
                        print(self.cuurrent_char)
                        self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                        
                self.prev_encode_action = False
                self.prev_check = False

            
            

            new_cuurrent_char[self.tokenizer.decode(tokens[:-1])]  = self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]
            curr_pos_tag = pos_tag(word_tokenize(text))
            if len(next_output)==1  or next_output[0]!="▁" or len(tokens)==0 or len(curr_pos_tag)==0:
                output_score[b_idx] = scores[b_idx]
                continue
            if len(tokens)!=0:
                curr_word ,current_tag = curr_pos_tag[-1]
            else:
                current_tag = None
            flag = 0
            for allowed_tag in self.allowed_pos_tag:
                if allowed_tag in current_tag:
                    flag=1
                    break
            if not flag:
                output_score[b_idx] = scores[b_idx]
                continue
            self.current_ids = torch.split(self._get_greenlist_ids((tokens)), self.table_size//(2**self.num_bit))
            
            secret_character = self.bit_stream[(self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]
            allowed_index = self.current_ids[bits2int(secret_character)]
            mask_tokens = torch.zeros_like(scores[b_idx])
            
            self.prev_encode_action = True
            if self.hard_encode:
                mask_tokens[allowed_index] = 1
                mask_tokens = mask_tokens.bool()
                mask_bad_tokens = torch.logical_not(mask_tokens)
                scores[b_idx] = scores[b_idx].masked_fill(mask_bad_tokens, -float("inf"))
            else:
                mask_tokens[allowed_index] = 10
                scores[b_idx]+=mask_tokens
            output_score[b_idx] = scores[b_idx]
        self.cuurrent_char = new_cuurrent_char
        return output_score.to(input_ids.device)
    def decode(self,text_output):
        m = []
        tokens = self.tokenizer.encode(text_output)
        sequence_counter = 0
        for i in range(len(tokens)):
            next_output = self.tokenizer.convert_ids_to_tokens(tokens[i])
            prev_tokens = tokens[:i]
            if (next_output[0]=="▁"or next_output in self.new_line) and len(prev_tokens)>0:
                inner_tokens = word_tokenize(self.tokenizer.decode(prev_tokens))
                prev_word , current_tag = pos_tag(inner_tokens)[-1]
                #print(pos_tag(inner_tokens))
                flag = 0
                for allowed_tag in self.allowed_pos_tag:
                    if allowed_tag in current_tag:
                        flag=1
                        break
                if not flag or next_output == "▁":
                    continue
                
                #print(prev_word,current_tag)
                if next_output[1].isalpha():
                    self.current_ids = torch.split(self._get_greenlist_ids(torch.tensor(prev_tokens).to(self.device)), self.table_size//(2**self.num_bit))
                    encoded_bit = (2**self.num_bit)-1
                    for k in range (2**self.num_bit):
                        if tokens[i] in self.current_ids[k]:
                            encoded_bit = k
                            break
                    decoded_bit = int2bits(encoded_bit,self.num_bit)
                    #if sequence_counter==len(self.bit_stream)//self.num_bit:
                    #    decoded_bit = decoded_bit[:len(self.bit_stream)%self.num_bit]
                    #    sequence_counter=0
                    #else:
                    #    sequence_counter+=1
                    m.extend(decoded_bit)
        ba = bitarray.bitarray(m)
        return ba
    

class SparseV2RandomNormalHashWatermarkDetector(SparseV2RandomNormalHashLogitsProcessor):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self,tokenizer,secret_watermark,threshold=0.2,**kwargs):
        super().__init__(tokenizer,secret_watermark,**kwargs )
        self.init_table()
        #self.based_bitstream = "".join(list(map(str, self.bit_stream))[:-(len(self.bit_stream)%8)])
        self.based_bitstream = "".join(list(map(str, self.bit_stream)))
        print(self.based_bitstream)
        self.threshold = threshold
    def detect(self,text: str = None,ret_value: bool = False):
        bit_stream = self.decode(text).to01()
        if len(bit_stream)==0:
            return False
        print(bit_stream)
        print("".join(list(self.based_bitstream*(ceil(len(bit_stream)/len(self.based_bitstream))))[:len(bit_stream)]))
        sample_bit = "".join(list(self.based_bitstream*(ceil(len(bit_stream)/len(self.based_bitstream))))[:len(bit_stream)])
        #diff = distance(sample_bit,bit_stream)
        #sample_bit = "".join(list(self.based_bitstream*(ceil(len(bit_stream)/len(self.based_bitstream))))[:len(bit_stream)-(diff//2)])
        result = distance(sample_bit,bit_stream)/len(bit_stream)
        print(result)
        if ret_value:
            result
        return result<self.threshold
            