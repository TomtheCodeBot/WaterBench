# Copyright 2024 THU-BPM MarkLLM.
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

# ================================================
# dip.py
# Description: Implementation of DiPmark algorithm
# ================================================

import torch
import hashlib
import random
from typing import Tuple, Union
import torch.nn.functional as F
from math import sqrt
from functools import partial
from transformers import LogitsProcessor, LogitsProcessorList


class DIPConfig:
    """Config class for DiP algorithm, load config file and initialize parameters."""

    def __init__(self,tokenizer,device, vocab: list[int] = None,
        gamma: float = 0.25,
        alpha: float = 0.1,
        seeding_scheme: str = "selfhash", *args, **kwargs):
        """
            Initialize the DiP configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """

        random.seed(42)
        hash_key = random.getrandbits(1024).to_bytes(128, "big")
        self.hash_key = hash_key
        
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_history_generation = 1
        self.ignore_history_detection = 1
        self.prefix_length = 5
        
        self.vocab_size = len(vocab)
        self.device = device
        self.generation_tokenizer = tokenizer
        #self.vocab_size = 32064
        self.z_threshold = 4
        

class DIPUtils:
    """Utility class for DiP algorithm, contains helper functions."""

    def __init__(self, config: DIPConfig,state_indicator=0, *args, **kwargs):
        """
            Initialize the DIP utility class.

            Parameters:
                config (DIPConfig): Configuration for the DiP algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.cc_history = set()
        self.state_indicator = state_indicator # 0 for generation, 1 for detection and visualization
        

    def _get_rng_seed(self, context_code: any) -> int:
        """Get the random seed from the given context code and private key."""
        if (
            (not self.config.ignore_history_generation and self.state_indicator == 0) or 
            (not self.config.ignore_history_detection and self.state_indicator == 1)
        ):
            self.cc_history.add(context_code)
            
        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.config.hash_key)
        
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed
    
    def _extract_context_code(self, context: torch.LongTensor) -> bytes:
        """Extract context code from the given context."""
        if self.config.prefix_length == 0:
            return context.detach().cpu().numpy().tobytes()
        else:
            return context[-self.config.prefix_length : ].detach().cpu().numpy().tobytes()
    
    def from_random(self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int) -> torch.LongTensor:
        """Generate a permutation from the random number generator."""
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        print(shuffle)

        return shuffle

    def reweight_logits(self, shuffle: torch.LongTensor, p_logits: torch.FloatTensor) -> torch.FloatTensor:
        """Reweight the logits using the shuffle and alpha."""
        unshuffle = torch.argsort(shuffle, dim=-1)
        
        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        
        # normalize the log_cumsum to force the last element to be 0
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        boundary_1 = torch.argmax((s_cumsum > self.config.alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - self.config.alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > self.config.alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax((s_cumsum > (1-self.config.alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (torch.gather(s_cumsum, -1, boundary_2) - (1-self.config.alpha)) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1-self.config.alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2/2 + s_all_portion_in_right_1/2
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, unshuffle)

        return p_logits + shift_logits
    
    def get_seed_for_cipher(self, input_ids: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Get the mask and seeds for the cipher."""
        
        batch_size = input_ids.size(0)
        context_codes = [
            self._extract_context_code(input_ids[i]) for i in range(batch_size)
        ]
        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self._get_rng_seed(context_code))
                for context_code in context_codes 
            ]
        )
        print(seeds)
        return mask, seeds
    
    def _get_green_token_quantile(self, input_ids: torch.LongTensor, vocab_size, current_token):
        """Get the vocab quantile of current token"""
        
        mask, seeds = self.get_seed_for_cipher(input_ids.unsqueeze(0))
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        
        mask = torch.tensor(mask, device=input_ids.device)
        print(vocab_size)
        shuffle = self.from_random(
            rng, vocab_size
        )
        token_quantile = [(torch.where(shuffle[0] == current_token)[0] +1)/vocab_size]
        return token_quantile, mask
    
    def _get_dip_score(self, input_ids: torch.LongTensor, vocab_size):
        """Get the DiP score of the input_ids"""
        scores = torch.zeros(input_ids.shape, device=input_ids.device)
        print("asdasdas")
        print(input_ids)
        for i in range(input_ids.shape[-1] - 1):
            pre = input_ids[ : i+1]
            cur = input_ids[i+1]
            token_quantile, mask = self._get_green_token_quantile(pre, vocab_size, cur)
            # if the current token is in the history and ignore_history_detection is False, set the score to -1
            if not self.config.ignore_history_detection and mask[0]: 
                scores[i + 1] = -1
            else:
                scores[i + 1] = torch.stack(token_quantile).reshape(-1)
        
        return scores
    
    def score_sequence(self, input_ids: torch.LongTensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        score = self._get_dip_score(input_ids, self.config.vocab_size)
        print(score)
        green_tokens = torch.sum(score >= self.config.gamma, dim=-1, keepdim=False)
        green_token_flags = torch.zeros_like(score)
        condition_indices = torch.nonzero(score >= self.config.gamma, as_tuple=False).reshape(-1)
        green_token_flags[condition_indices] = 1
        green_token_flags[:self.config.prefix_length] = -1
        
        # Use two different ways to calculate z_score depending on whether to ignore history
        if not self.config.ignore_history_detection:
            ignored_indices = torch.nonzero(score == -1, as_tuple=False).reshape(-1)
            
            # Visualize the ignored tokens as ignored
            green_token_flags[ignored_indices] = -1
            
            # Calculate z_score using the sequence length after ignoring the ignored tokens
            sequence_length_for_calculation = input_ids.size(-1) - ignored_indices.size(0)
            z_score = (green_tokens - (1-self.config.gamma) * sequence_length_for_calculation) / sqrt(sequence_length_for_calculation)
        else:
            z_score = (green_tokens - (1-self.config.gamma) * input_ids.size(-1)) / sqrt(input_ids.size(-1))
        
        return z_score.item(), green_token_flags.tolist()


class DIPLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for DiP algorithm, process logits to add watermark."""

    def __init__(self, config: DIPConfig, utils: DIPUtils, *args, **kwargs):
        """
            Initialize the DIP logits processor.

            Parameters:
                config (DIPConfig): Configuration for the DiP algorithm.
                utils (DIPUtils): Utility class for the DiP algorithm.
        """
        self.config = config
        self.utils = utils
        self.prompt_slice = 0
    def _apply_watermark(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Apply watermark to the scores."""
        mask, seeds = self.utils.get_seed_for_cipher(input_ids)
        
        rng = [
            torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds
        ]
        mask = torch.tensor(mask, device=scores.device)
        print(scores.shape)
        shuffle = self.utils.from_random(
            rng, scores.size(1)
        )

        reweighted_scores = self.utils.reweight_logits(shuffle, scores)
        
        return mask, reweighted_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        sliced_input = input_ids[0][self.prompt_slice:]
        sliced_input = torch.unsqueeze(sliced_input,0)
        if sliced_input.shape[-1] < self.config.prefix_length:
            return scores
        
        mask, reweighted_scores = self._apply_watermark(sliced_input, scores)
        print(reweighted_scores - scores)
        if self.config.ignore_history_generation:
            return reweighted_scores
        else:
            return torch.where(mask[:, None], scores, reweighted_scores)
class DIPDetector:
    def __init__(self, config: DIPConfig, utils: DIPUtils) -> None:
        """
            Initialize the DIP logits processor.

            Parameters:
                config (DIPConfig): Configuration for the DiP algorithm.
                utils (DIPUtils): Utility class for the DiP algorithm.
        """
        self.config = config
        self.utils = utils
    def detect(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""
        
        # Set the state indicator to 1 for detection
        self.utils.state_indicator = 1
        print(text)
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z-score using a utility method
        z_score, _ = self.utils.score_sequence(encoded_text)
        
        # Determine if the z-score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Clear the history
        self.utils.cc_history.clear()

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "z_score": z_score}
        else:
            return (is_watermarked, z_score)
