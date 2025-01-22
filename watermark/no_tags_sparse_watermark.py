
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

from nltk.tokenize import WhitespaceTokenizer
span_tokenize = WhitespaceTokenizer().span_tokenize

class NoTagSparseWatermark(LogitsProcessor):
    def __init__(self,tokenizer,prompt_slice,gamma=0.5,delta=10,allowed_pos_tag=["V"],index = 0,seeding_scheme="lefthash",hard_encode=True,modular = 3):
        self.gamma = gamma
        self.delta = delta
        self.tokenizer = tokenizer
        self.input_index = index
        self.cuurrent_char = None
        self.prompt_slice = prompt_slice
        self.prev_encode_action = False
        self.hard_encode = hard_encode
        self.last_input = []
        self.new_line = ["<0x0A>"]
        self.prev_check = False
        self.device = "cuda"
        self.vocab_size = len(tokenizer)
        self.rng = torch.Generator(device=self.device)
        self.all_observed = 0
        self._initialize_seeding_scheme(seeding_scheme)
        self.current_prf = None
        self.modular = modular
        print(self.modular)
    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG.")

        prf_key = prf_lookup[self.prf_type](input_ids[-self.context_width :], salt_key=self.hash_key)
        self.current_prf = prf_key
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]  # new
        return greenlist_ids
    
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)

        green_tokens_mask[torch.tensor(greenlist_token_ids)]=1
        return green_tokens_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        for b_idx, input_seq in enumerate(input_ids):
            if self.self_salt:
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq) 

            # logic for computing and storing spike entropies for analysis
            if self.current_prf is not None and self.current_prf%self.modular==0:
                green_tokens_mask = self._calc_greenlist_mask(scores=scores[b_idx], greenlist_token_ids=greenlist_ids)
                green_tokens_mask = green_tokens_mask.bool()
                mask_bad_tokens = torch.logical_not(green_tokens_mask)
                scores[b_idx] = scores[b_idx].masked_fill(mask_bad_tokens, -float("inf"))

        return scores
    def decode(self,text_output):
        m = []
        tokens = self.tokenizer.encode(text_output)
        self.current_prf = None
        for i in range(len(tokens)):
            prev_tokens = tokens[:i]
            if len(prev_tokens) > 0:
                ids = self._get_greenlist_ids(torch.tensor(prev_tokens).to(self.device))
                if self.current_prf is not None and self.current_prf%self.modular==0: 
                    if tokens[i] in ids:
                        m.extend("1")
                    else:
                        m.extend("0")
        return m
    
class NoTagSparseDetector(NoTagSparseWatermark):
    def __init__(self, tokenizer, prompt_slice, gamma=0.5, delta=10, allowed_pos_tag=["V"], index=0, seeding_scheme="lefthash",hard_encode=True,modular = 3):
        super().__init__(tokenizer, prompt_slice, gamma, delta, allowed_pos_tag, index, seeding_scheme,hard_encode,modular)
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
        print("z", z)
        return z
    def detect(self,
               text_output,
               return_scores: bool = True,):
        list_green = self.decode(text_output)
        
        z_score = self._compute_z_score(list_green.count("1"), len(list_green))
        # print("z_score is:", z_score)
        return z_score