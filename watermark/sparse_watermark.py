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
class StegoLogitsProcessor(LogitsProcessor):
    def __init__(self,tokenizer,secret_watermark,prompt_slice,bitnum=3,allowed_pos_tag=["V"],granularity = "word",gap = 5,index = 0,shifted = 0):
        self.granularity = granularity
        self.gap = gap
        self.tokenizer = tokenizer
        self.input_index = index
        self.shift_first_token = shifted
        self.secret_watermark = secret_watermark
        self.ba = bitarray.bitarray()
        self.ba.frombytes(self.secret_watermark.encode('utf-8'))
        self.bit_stream = self.ba.tolist()
        self.num_bit = bitnum
        self.bit_stream.extend(["0"for _ in range(len(self.bit_stream)%self.num_bit)])
        self.cuurrent_char = None
        self.prompt_slice = prompt_slice
        self.allowed_pos_tag = allowed_pos_tag
        self.last_word_scores = None
        self.prev_encode_action = False
        self.ending_line = False
        self.hard_encode = True
        self.last_input = []
        self.new_line = ["<0x0A>"]
    def init_table(self):
        self.table = {}
        for i in range(len(self.tokenizer.get_vocab().keys())):
            if self.tokenizer.convert_ids_to_tokens(i)[0] != "▁" or len(self.tokenizer.convert_ids_to_tokens(i))==1 or not self.tokenizer.convert_ids_to_tokens(i)[1].isalpha()  :
                continue
            if self.tokenizer.convert_ids_to_tokens(i)[1].lower() not in self.table.keys():
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()] = [i]
            else:
                self.table[self.tokenizer.convert_ids_to_tokens(i)[1].lower()].append(i)
        list_keys = list(self.table.keys())
        list_keys.sort()
        self.table_bit = [[] for _ in range(2**self.num_bit)]
        self.reverse_bit = {}
        for i in range(len(list_keys)):
            self.table_bit[i%(2**self.num_bit)].extend(self.table[list_keys[i]])
            self.reverse_bit[list_keys[i]]=i%(2**self.num_bit)
    def two__call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_score, next_token = torch.sort(scores, dim=1,descending=True)
        next_token = next_token[:,0]
        output_score = torch.zeros(scores.shape)
        new_cuurrent_char = {}
        for b_idx in range(input_ids.shape[0]):
            token_added = torch.cat([input_ids[b_idx], next_token[b_idx].reshape(1)], dim=-1)
            tokens = input_ids[b_idx]
            if self.cuurrent_char is None:
                self.cuurrent_char={}
                self.cuurrent_char[self.tokenizer.decode(tokens[:-2])] = 0
            text = self.tokenizer.decode(token_added)
            
            next_output = self.tokenizer.convert_ids_to_tokens(next_token[b_idx].item())
            
            curr_word ,current_tag = pos_tag(word_tokenize(text))[-1]
            
            if next_output[0]=="▁" or next_output in self.new_line:
                if self.prev_encode_action:
                    secret_character = self.bit_stream[(self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]

                    inner_tokens = word_tokenize(self.tokenizer.decode(tokens))
                    prev_word , check_tag = pos_tag(inner_tokens)[-1]
                    if prev_word[0].lower()not in punctuation:
                        
                        encoded_bit = self.reverse_bit[prev_word[0].lower()]
                        flag=0
                        for allowed_tag in self.allowed_pos_tag:
                            if allowed_tag in check_tag:
                                flag=1
                                break
                        
                        if  flag:
                            if self.hard_encode:
                                if encoded_bit == bits2int(secret_character):
                                    
                                    self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                                else:
                                    print(prev_word,encoded_bit,bits2int(secret_character),check_tag)
                                    raise Exception
                            else:
                                self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                    self.prev_encode_action = False
                
            new_cuurrent_char[self.tokenizer.decode(tokens[:-1])]  = self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]
            if len(next_output)==1  or next_output[0]!="▁":
                output_score[b_idx] = scores[b_idx]
                continue
                    
            flag = 0
            for allowed_tag in self.allowed_pos_tag:
                if allowed_tag in current_tag:
                    flag=1
                    break
            if not flag:
                output_score[b_idx] = scores[b_idx]
                continue

            secret_character = self.bit_stream[(self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]
            allowed_index = torch.tensor(self.table_bit[bits2int(secret_character)])
            mask_tokens = torch.zeros_like(scores[b_idx])*(1e-5)
            
            self.prev_encode_action = True
            if self.hard_encode:
                mask_tokens[allowed_index] = 1
                scores[b_idx]*=mask_tokens
            else:
                mask_tokens[allowed_index] = 10
                scores[b_idx]+=mask_tokens
            output_score[b_idx] = scores[b_idx]
        self.cuurrent_char = new_cuurrent_char
        return output_score.to(input_ids.device)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_score, next_token = torch.sort(scores, dim=1,descending=True)
        next_token = next_token[:,0]
        output_score = torch.zeros(scores.shape)
        new_cuurrent_char = {}
        for b_idx in range(input_ids.shape[0]):
            sliced_input = input_ids[b_idx][self.prompt_slice:]
            token_added = torch.cat([sliced_input, next_token[b_idx].reshape(1)], dim=-1)
            tokens = sliced_input
            if self.cuurrent_char is None:
                self.cuurrent_char={}
                self.cuurrent_char[self.tokenizer.decode(tokens[:-2])] = 0
            text = self.tokenizer.decode(token_added)
            
            next_output = self.tokenizer.convert_ids_to_tokens(next_token[b_idx].item())
            if next_output[0]=="▁" or next_output in self.new_line or next_output[0] in punctuation:
                if self.prev_encode_action:
                    secret_character = self.bit_stream[(self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]

                    inner_tokens = word_tokenize(self.tokenizer.decode(tokens))
                    if inner_tokens[-1][0].lower() in punctuation:

                        prev_word , check_tag = pos_tag(inner_tokens)[-2]
                    else:
                        prev_word , check_tag = pos_tag(inner_tokens)[-1]
                    if not (prev_word[0].isnumeric() or prev_word[0] in punctuation):
                        encoded_bit = self.reverse_bit[prev_word[0].lower()]
                        flag=0
                        for allowed_tag in self.allowed_pos_tag:
                            if allowed_tag in check_tag:
                                flag=1
                                break
                        
                        if  flag:
                            if self.hard_encode:
                                if encoded_bit == bits2int(secret_character):
                                    self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                                    self.last_input.append(prev_word.lower())
                                else:
                                    print(prev_word,encoded_bit,bits2int(secret_character))
                                    #raise Exception
                            else:
                                print(self.cuurrent_char)
                                self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]+=self.num_bit
                            
                    self.prev_encode_action = False
            
            if len(tokens)!=0:
                curr_word ,current_tag = pos_tag(word_tokenize(text))[-1]
                
            new_cuurrent_char[self.tokenizer.decode(tokens[:-1])]  = self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]
            if len(next_output)==1  or next_output[0]!="▁":
                output_score[b_idx] = scores[b_idx]
                continue
                    
            flag = 0
            for allowed_tag in self.allowed_pos_tag:
                if allowed_tag in current_tag:
                    flag=1
                    break
            if not flag:
                output_score[b_idx] = scores[b_idx]
                continue
            secret_character = self.bit_stream[(self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream)):((self.cuurrent_char[self.tokenizer.decode(tokens[:-2])]%len(self.bit_stream))+self.num_bit)]
            allowed_index = torch.tensor(self.table_bit[bits2int(secret_character)])
            mask_tokens = torch.zeros_like(scores[b_idx])*(1e-5)
            
            self.prev_encode_action = True
            if self.hard_encode:
                mask_tokens[allowed_index] = 1
                scores[b_idx]*=mask_tokens
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
        for i in range(1,len(tokens)):
            next_output = self.tokenizer.convert_ids_to_tokens(tokens[i])
            prev_tokens = tokens[:i]
            # if next_output[0]=="▁" and len(prev_tokens)>0 :

            if (next_output[0]=="▁" and len(prev_tokens)>0) or next_output in self.new_line or next_output[0] in punctuation:
                inner_tokens = word_tokenize(self.tokenizer.decode(prev_tokens))
                if inner_tokens[-1][0].lower() in punctuation:

                    prev_word , current_tag = pos_tag(inner_tokens)[-2]
                else:
                    prev_word , current_tag = pos_tag(inner_tokens)[-1]
                
                if prev_word[0].isalpha():
                    flag = 0
                    for allowed_tag in self.allowed_pos_tag:
                        if allowed_tag in current_tag:
                            flag=1
                            break
                    if not flag:
                        continue
                    #print(prev_word)
                    encoded_bit = self.reverse_bit[prev_word[0].lower()]
                    decoded_bit = int2bits(encoded_bit,self.num_bit)
                    if sequence_counter==len(self.bit_stream)//self.num_bit:
                        decoded_bit = decoded_bit[:len(self.bit_stream)%self.num_bit]
                        sequence_counter=0
                    else:
                        sequence_counter+=1
                    m.extend(decoded_bit)

        #print(self.last_input)
        ba = bitarray.bitarray(m)
        
        return ba
    

class StegoWatermarkDetector(StegoLogitsProcessor):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self,tokenizer,secret_watermark,prompt_slice=None,bitnum=3,allowed_pos_tag=["V"],granularity = "word",gap = 5,index = 0,shifted = 0,threshold=0.29):
        super().__init__(tokenizer,secret_watermark,prompt_slice,bitnum,allowed_pos_tag,granularity,gap ,index,shifted )
        self.init_table()
        self.based_bitstream = "".join(list(map(str, self.bit_stream))[:-(len(self.bit_stream)%8)])
        print(self.based_bitstream)
        self.threshold = threshold
    def detect(self,text: str = None,ret_value: bool = False):
        bit_stream = self.decode(text).to01()
        if len(bit_stream)==0:
            return False
        print(bit_stream)
        print("".join(list(self.based_bitstream*(ceil(len(bit_stream)/len(self.based_bitstream))))[:len(bit_stream)]))
        sample_bit = "".join(list(self.based_bitstream*(ceil(len(bit_stream)/len(self.based_bitstream))))[:len(bit_stream)])
        diff = distance(sample_bit,bit_stream)
        sample_bit = "".join(list(self.based_bitstream*(ceil(len(bit_stream)/len(self.based_bitstream))))[:len(bit_stream)-(diff//2)])
        result = distance(sample_bit,bit_stream)/len(bit_stream)
        print(result)
        if ret_value:
            result
        return result<self.threshold
            