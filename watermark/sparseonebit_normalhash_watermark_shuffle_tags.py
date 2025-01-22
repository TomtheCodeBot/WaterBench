
from transformers import AutoTokenizer
import torch
import fastchat.model
import bitarray
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser
from string import punctuation
import time
from .additional_utils.alternative_prf_schemes import prf_lookup, seeding_scheme_lookup
from transformers import LogitsProcessor
import math
import copy
from nltk.tokenize import WhitespaceTokenizer
import string
span_tokenize = WhitespaceTokenizer().span_tokenize

ENGLISH_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
UNIVERSAL_TAGS = [
    "DET",  # Determiner
    "NOUN", # Noun
    "VERB", # Verb
]
class SparseOneBitNormalHashRandomTag(LogitsProcessor):
    def __init__(self,tokenizer,prompt_slice,gamma=0.5,delta=10,allowed_pos_tag=["V"],index = 0,seeding_scheme="lefthash",hard_encode=True):
        self.gamma = gamma
        self.delta = delta
        self.tokenizer = tokenizer
        self.input_index = index
        self.cuurrent_char = None
        self.prompt_slice = prompt_slice
        random_idx = torch.randint(low=0, high=len(UNIVERSAL_TAGS), size=(1,)).item()
        self.allowed_pos_tag  = UNIVERSAL_TAGS[random_idx]

        self.hard_encode = hard_encode
        self.last_input = []
        self.new_line = ["<0x0A>"]
        self.prev_check = False
        self.device = "cuda"
        self.rng = torch.Generator(device=self.device)
        self.all_observed = 0
        self._initialize_seeding_scheme(seeding_scheme)
        self.prev_ids = []
    def init_table(self):
        self.table = []
        for i in range(len(self.tokenizer.get_vocab().keys())):
            if self.tokenizer.convert_ids_to_tokens(i)[0] != "▁" or len(self.tokenizer.convert_ids_to_tokens(i))==1 or not self.tokenizer.convert_ids_to_tokens(i)[1].lower() in ENGLISH_ALPHABET  :
                continue
            self.table.append(i)
        self.table_size = len(self.table)
        self.table = torch.tensor(self.table).to(self.device)
        return

    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG.")
        print(f"Hashing:{input_ids[-self.context_width :]}")
        prf_key = prf_lookup[self.prf_type](input_ids[-self.context_width :], salt_key=self.hash_key)
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)

        greenlist_size = int(self.table_size * self.gamma)
        vocab_permutation = torch.randperm(self.table_size, device=input_ids.device, generator=self.rng)
        vocab_permutation = self.table[vocab_permutation]
        greenlist_ids = vocab_permutation[:greenlist_size]  # new
        
        return greenlist_ids
    
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)

        green_tokens_mask[torch.tensor(greenlist_token_ids)]=1
        return green_tokens_mask

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
            print(text)
            print(input_ids[b_idx][-1])
            print(input_ids[b_idx][-1] in self.prev_ids)
            #print(pos_tag(word_tokenize(text)))
            next_output = self.tokenizer.convert_ids_to_tokens(next_token[b_idx].item())
            
            
            curr_pos_tag = pos_tag(word_tokenize(text),tagset='universal')
            if len(curr_pos_tag)>0:
                print(next_output,curr_pos_tag[-1])
            print((len(next_output)==1 and next_output not in string.punctuation)  or next_output[0]!="▁" or len(tokens)==0 or len(curr_pos_tag)==0)
            #if len(next_output)==1  or next_output[0]!="▁" or len(tokens)==0 or len(curr_pos_tag)==0:
            if (len(next_output)==1 and next_output not in string.punctuation)  or {next_output[0]!="▁"and next_output not in string.punctuation} or len(tokens)==0 or len(curr_pos_tag)==0:
                output_score[b_idx] = scores[b_idx]
                continue
            if len(tokens)!=0:
                curr_word ,current_tag = curr_pos_tag[-1]
            flag = 0
            if (curr_word in ['.', '!', '?'] or (current_tag == '.' and curr_word != ',')):
                print(curr_word ,current_tag)
                random_idx = torch.randint(low=0, high=len(UNIVERSAL_TAGS), size=(1,)).item()
                self.allowed_pos_tag  = UNIVERSAL_TAGS[random_idx]
                print("Changed to ",self.allowed_pos_tag)
            print(f"before passing through allows tags:{curr_word ,current_tag}")
            for allowed_tag in self.allowed_pos_tag:
                if allowed_tag in current_tag[:len(allowed_tag)]:
                    flag=1
                    break
            if not flag :
                output_score[b_idx] = scores[b_idx]
                continue
            print(curr_word ,current_tag)
            #ids = self._get_greenlist_ids(self.tokenizer.encode(text,return_tensors="pt").to(self.device))
            ids = self._get_greenlist_ids(tokens)
            self.prev_ids = ids
            mask_tokens = self._calc_greenlist_mask(scores[b_idx],ids)
            print(f"Encoding status:{self.hard_encode}")
            if self.hard_encode:
                mask_tokens = mask_tokens.bool()
                mask_bad_tokens = torch.logical_not(mask_tokens)
                scores[b_idx] = scores[b_idx].masked_fill(mask_bad_tokens, -float("inf"))
            else:
                mask_tokens = mask_tokens.float()
                mask_tokens *= self.delta
                scores[b_idx]+=mask_tokens
            output_score[b_idx] = scores[b_idx]
        return output_score.to(input_ids.device)
    def decode(self,text_output):
        m = []
        tokens = self.tokenizer.encode(text_output)
        sequence_counter = 0
        next_outputs = self.tokenizer.convert_ids_to_tokens(tokens)
        slots = {
            "DET":0,  # Determiner
            "NOUN":0, # Noun
            "VERB":0, # Verb
        }
        green_tokens = copy.deepcopy(slots)
        all_tokens = copy.deepcopy(slots)
        for i in range(len(tokens)):
            next_output = next_outputs[i]
            prev_tokens = tokens[:i]
            if (next_output[0]=="▁" and next_output not in self.new_line) and len(prev_tokens)>0 and len(next_output)>1:
                
                inner_tokens = word_tokenize(self.tokenizer.decode(prev_tokens))
                prev_word , current_tag = pos_tag(inner_tokens,tagset="universal")[-1]
                if current_tag in slots.keys():
                    print(prev_word , current_tag)
                print(prev_word , current_tag)
                if (prev_word in ['.', '!', '?']  or( current_tag == '.' and prev_word != ',')):
                    print("new sentence")
                    print(self.tokenizer.decode(prev_tokens))
                    highest_portions = -1
                    highest_key = None
                    for key in slots.keys():
                        print(green_tokens[key],all_tokens[key])
                        if all_tokens[key]==0:
                            continue
                        
                        if green_tokens[key]/all_tokens[key]>highest_portions:
                            highest_portions = green_tokens[key]/all_tokens[key]
                            highest_key = key
                    if highest_portions != -1:
                        green_list_batch = ["0"]*all_tokens[highest_key]
                        green_list_batch[:green_tokens[highest_key]] = "1"
                        m.extend(copy.deepcopy(green_list_batch))
                    
                    green_tokens = copy.deepcopy(slots)
                    all_tokens = copy.deepcopy(slots)
                if current_tag in slots.keys():
                    ids = self._get_greenlist_ids(torch.tensor(prev_tokens).to(self.device))
                    if current_tag in slots.keys():
                        print(tokens[i])
                        print(tokens[i] in ids)
                        if tokens[i] in ids :
                            green_tokens[current_tag]+=1
                            print(f"{current_tag} = {green_tokens[current_tag]}")
                        all_tokens[current_tag]+=1
        return m
    # def decode(self,text_output):
    #     m = []
    #     all_words_pos = word_tokenize(text_output)
    #     all_span =  [span for span in span_tokenize(text_output)]
    #     all_tags = pos_tag(all_words_pos,tagset='universal')
    #     tokens = self.tokenizer.encode(text_output)
    #     sequence_counter = 0
    #     next_outputs = self.tokenizer.convert_ids_to_tokens(tokens)
    #     slots = {
    #         "ADJ":0,  # Adjective
    #         "ADP":0,  # Adposition (prepositions/postpositions)
    #         "ADV":0,  # Adverb
    #         "CONJ":0, # Conjunction
    #         "DET":0,  # Determiner
    #         "NOUN":0, # Noun
    #         "NUM":0,  # Numeral
    #         "PRON":0, # Pronoun
    #         "PRT":0,  # Particle
    #         "VERB":0, # Verb
    #         ".":0,    # Punctuation
    #         "X":0     # Other (foreign words, typos, abbreviations)
    #     }
    #     green_tokens = copy.deepcopy(slots)
    #     all_tokens = copy.deepcopy(slots)
    #     for i in range(len(all_words_pos)-1):
            
                
    #         if (all_tags[i][0] in ['.', '!', '?'] or all_tags[i][1] == '.'):
    #             highest_portions = -1
    #             highest_key = None
    #             for key in slots.keys():
    #                 if all_tokens[key]==0:
    #                     continue
    #                 if green_tokens[key]/all_tokens[key]>highest_portions:
    #                     highest_portions = green_tokens[key]/all_tokens[key]
    #                     highest_key = key
    #             green_list_batch = ["0"]*all_tokens[highest_key]
    #             green_list_batch[:green_tokens[highest_key]] = "1"
    #             m.extend(copy.deepcopy(green_list_batch))
                
    #             green_tokens = copy.deepcopy(slots)
    #             all_tokens = copy.deepcopy(slots)

    #         ids = self._get_greenlist_ids(torch.tensor(self.tokenizer.encode(text_output[:all_span[i][-1]])).to(self.device))
    #         if self.tokenizer.encode(all_tags[i+1][0])[0] in ids:

    #             green_tokens[all_tags[i][1]]+=1
    #         all_tokens[all_tags[i][1]]+=1
    #     return m

class SparseOneBitNormalHashRandomTagDetector(SparseOneBitNormalHashRandomTag):
    def __init__(self, tokenizer, prompt_slice, gamma=0.5, delta=10, allowed_pos_tag=["V"], index=0, seeding_scheme="lefthash",hard_encode=True):
        super().__init__(tokenizer, prompt_slice, gamma, delta, allowed_pos_tag, index, seeding_scheme,hard_encode)
        self.init_table()
    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        print(self.gamma)
        print("observed_count", observed_count)
        print("T", T)
        self.all_observed+=observed_count
        numer = observed_count - expected_count * T
        denom = math.sqrt(T * expected_count * (1 - expected_count))
        try:
            z = numer / denom
        except ZeroDivisionError:
            return 0
        return z
    def detect(self,
               text_output,
               return_scores: bool = True,):
        list_green = self.decode(text_output)
        
        z_score = self._compute_z_score(list_green.count("1"), len(list_green))
        # print("z_score is:", z_score)
        return z_score