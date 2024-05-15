from tqdm import tqdm
import os
import torch.nn.functional as F
from watermark.old_watermark import BlacklistLogitsProcessor
from watermark.our_watermark import OurBlacklistLogitsProcessor
from watermark.gptwm import GPTWatermarkLogitsWarper
from watermark.watermark_v2 import WatermarkLogitsProcessor
from watermark.sparse_watermark import StegoLogitsProcessor,StegoWatermarkDetector
from watermark.sparsev2_watermark import SparseV2LogitsProcessor,SparseV2WatermarkDetector
from watermark.og_watermark import OGWatermarkLogitsProcessor
from watermark.sparse_one_bit_watermark import SparseOneBit,SparseOneBitDetector
from watermark.sparsev2seeded_watermark import SparseV2RandomLogitsProcessor,SparseV2RandomWatermarkDetector
from watermark.sparsev2seedednormalhash_watermark import SparseV2RandomNormalHashLogitsProcessor,SparseV2RandomNormalHashWatermarkDetector
from watermark.sparseonebit_normalhash_watermark import SparseOneBitNormalHash,SparseOneBitNormalHashDetector
from watermark.entropy_checker import POSEntropyChecker
from transformers import  LogitsProcessorList
    
    
class Generator():
    def __init__(self, args, tokenizer, model) -> None:
        self.mode = args.mode # watermark mode
        self.init_seed, self.dyna_seed, self.gamma, \
        self.delta, self.bl_type, self.num_beams, self.sampling_temp = args.initial_seed, args.dynamic_seed, args.gamma, args.delta, args.bl_type, args.num_beams, args.sampling_temp
        self.tokenizer = tokenizer
        self.model = model # language model
        
        self.all_token_ids = list(tokenizer.get_vocab().values())
        self.vocab_size = len(self.all_token_ids)
        # if self.vocab_size != model.config.padded_vocab_size:
        #     self.vocab_size = model.config.padded_vocab_size
        
        self.bl_processor = BlacklistLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed)
        self.bl_processor.tokenizer = tokenizer
        self.logit_processor_lst = LogitsProcessorList([self.bl_processor])
        self.random_bit_string = args.random_bit_String
        if args.mode == 'og':
            self.bl_processor = OGWatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        seeding_scheme = "lefthash",  )
            self.logit_processor_lst = LogitsProcessorList([self.bl_processor])

        if args.mode == 'entropycheck':
            self.bl_processor = POSEntropyChecker(tokenizer=tokenizer )
            self.logit_processor_lst = LogitsProcessorList([self.bl_processor])
        if args.mode == 'ogv2':
            self.bl_processor = OGWatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        seeding_scheme = "selfhash",  )
            self.logit_processor_lst = LogitsProcessorList([self.bl_processor])
        if args.mode == 'new': 
            self.bl_processor = OurBlacklistLogitsProcessor(tokenizer=tokenizer,
                                        bad_words_ids=None, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        vocab=self.all_token_ids, 
                                        all_vocab_size=self.vocab_size, 
                                        bl_proportion=1-self.gamma,
                                        bl_logit_bias=self.delta,
                                        bl_type=self.bl_type, 
                                        initial_seed=self.init_seed, 
                                        dynamic_seed=self.dyna_seed)
            self.logit_processor_lst = LogitsProcessorList([self.bl_processor])
            
        if args.mode == 'gpt':
            
            watermark_processor = GPTWatermarkLogitsWarper(vocab_size=self.vocab_size,
                                                        fraction=args.gamma,
                                                        strength=args.delta)
            
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])   
            
        if args.mode == 'v2':
            watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        seeding_scheme=args.seeding_scheme,
                                                        select_green_tokens=args.select_green_tokens)
            self.logit_processor_lst = LogitsProcessorList([watermark_processor]) 
        if args.mode == 'sparse':
            watermark_processor = StegoLogitsProcessor(tokenizer=tokenizer,
                                                        secret_watermark = "password",
                                                        prompt_slice=None,
                                                        )
            self.detector = StegoWatermarkDetector(
                tokenizer = tokenizer,
                secret_watermark= "password")
            watermark_processor.init_table()
            self.logit_processor_lst = LogitsProcessorList([watermark_processor]) 
        if args.mode == 'sparsev2':
            watermark_processor = SparseV2LogitsProcessor(tokenizer=tokenizer,
                                                        secret_watermark = "password",
                                                        prompt_slice=None,
                                                        random_bit_string=self.random_bit_string
                                                        )
            self.detector = SparseV2WatermarkDetector(
                tokenizer = tokenizer,
                secret_watermark= "password",
                prompt_slice=None,
                random_bit_string=self.random_bit_string)
            watermark_processor.init_table()
            self.logit_processor_lst = LogitsProcessorList([watermark_processor]) 
        if args.mode == 'sparsev2seeded':
            watermark_processor = SparseV2RandomLogitsProcessor(tokenizer=tokenizer,
                                                        secret_watermark = "password",
                                                        prompt_slice=None,
                                                        random_bit_string=self.random_bit_string
                                                        )
            self.detector = SparseV2RandomWatermarkDetector(
                tokenizer = tokenizer,
                secret_watermark= "password",
                prompt_slice=None,
                random_bit_string=self.random_bit_string)
            watermark_processor.init_table()
            self.logit_processor_lst = LogitsProcessorList([watermark_processor]) 
        if args.mode == 'onebitsparse':
            watermark_processor = SparseOneBit(tokenizer=tokenizer,
                                               gamma=args.gamma,
                                                delta=args.delta,
                                                prompt_slice=None,
                                                hard_encode=True if self.bl_type=="hard" else False,
                                                allowed_pos_tag=args.pos_tag
                                                )
            print(f"[INFO]:{watermark_processor.hard_encode}")
            self.detector = SparseOneBitDetector(
                tokenizer = tokenizer,
                gamma=args.gamma,
                delta=args.delta,
                prompt_slice=None,
                hard_encode=True if self.bl_type=="hard" else False,
                allowed_pos_tag=args.pos_tag
                )
        if args.mode == 'sparsev2seedednormalhash':
            watermark_processor = SparseV2RandomNormalHashLogitsProcessor(tokenizer=tokenizer,
                                                        secret_watermark = "password",
                                                        prompt_slice=None,
                                                        random_bit_string=self.random_bit_string
                                                        )
            self.detector = SparseV2RandomNormalHashWatermarkDetector(
                tokenizer = tokenizer,
                secret_watermark= "password",
                prompt_slice=None,
                random_bit_string=self.random_bit_string)
            watermark_processor.init_table()
            self.logit_processor_lst = LogitsProcessorList([watermark_processor]) 
        if args.mode == 'onebitsparsenormalhash':
            watermark_processor = SparseOneBitNormalHash(tokenizer=tokenizer,
                                               gamma=args.gamma,
                                                delta=args.delta,
                                                prompt_slice=None,
                                                hard_encode=True if self.bl_type=="hard" else False,
                                                allowed_pos_tag=args.pos_tag
                                                )
            
            print(f"[INFO]:{watermark_processor.hard_encode}")
            self.detector = SparseOneBitNormalHashDetector(
                tokenizer = tokenizer,
                gamma=args.gamma,
                delta=args.delta,
                prompt_slice=None,
                hard_encode=True if self.bl_type=="hard" else False,
                allowed_pos_tag=args.pos_tag
                )
            watermark_processor.init_table()
            self.logit_processor_lst = LogitsProcessorList([watermark_processor]) 
            
    def generate(self, input_ids, max_new_tokens):
        if self.mode == 'new':
            example = {}
            
            outputs = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                logits_processor = self.logit_processor_lst,
                do_sample=True,
                top_k=0,
                temperature=self.sampling_temp
            )

            example.update({"bl_vocabularys":self.logit_processor_lst[0].get_and_clear_vocabularys()})
            # remove the attached input from output for some model
            scores = outputs.scores
            output_ids = outputs.sequences[0, -len(scores):]


            # compute logprob for each token
            completions_tokens = []
            completions_logprob = 0

            for score, token, vocabulary in zip(scores, output_ids, example["bl_vocabularys"], strict=True):
                logprobs = F.log_softmax(score[0], dim=-1)
                logprob = logprobs[token].item()
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                    'logprob': logprob,
                    'vocabulary': vocabulary,
                })
                completions_logprob += logprob
            
            completions_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return completions_text, completions_tokens
        else:    
        
            if self.mode == 'no':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                )

            elif self.mode == 'old':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    temperature=self.sampling_temp
                )

            elif self.mode == 'og':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    temperature=self.sampling_temp
                )
            elif self.mode == 'ogv2':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    temperature=self.sampling_temp
                )
            elif self.mode == 'gpt':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    top_p=0.9
                )

            elif self.mode == 'v2':
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    temperature=self.sampling_temp
                )
            elif self.mode == 'sparse':
                self.logit_processor_lst[0].prompt_slice = len(input_ids[0])
                self.logit_processor_lst[0].last_input = []
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                )
                
                self.logit_processor_lst[0].cuurrent_char = None
                self.logit_processor_lst[0].prev_encode_action = False
            elif self.mode == 'sparsev2':
                self.logit_processor_lst[0].prompt_slice = len(input_ids[0])
                self.logit_processor_lst[0].last_input = []
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                )
                
                self.logit_processor_lst[0].cuurrent_char = None
                self.logit_processor_lst[0].prev_encode_action = False
            elif self.mode == 'sparsev2seeded':
                self.logit_processor_lst[0].prompt_slice = len(input_ids[0])
                self.logit_processor_lst[0].last_input = []
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                )
                
                self.logit_processor_lst[0].cuurrent_char = None
                self.logit_processor_lst[0].prev_encode_action = False
            elif self.mode == 'sparsev2seedednormalhash':
                self.logit_processor_lst[0].prompt_slice = len(input_ids[0])
                self.logit_processor_lst[0].last_input = []
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                )
                
                self.logit_processor_lst[0].cuurrent_char = None
                self.logit_processor_lst[0].prev_encode_action = False
            elif self.mode == 'onebitsparse':
                
                self.logit_processor_lst[0].prompt_slice = len(input_ids[0])
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                )
            elif self.mode == 'onebitsparsenormalhash':
                
                self.logit_processor_lst[0].prompt_slice = len(input_ids[0])
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                )
            elif self.mode == 'entropycheck':
                
                self.logit_processor_lst[0].prompt_slice = len(input_ids[0])
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                )
                print(self.logit_processor_lst[0].entropy_per_tag)
            # remove the attached input from output for some model
            scores = outputs.scores
            output_ids = outputs.sequences[0, -len(scores):]

            # compute logprob for each token
            completions_tokens = []
            completions_logprob = 0

            for score, token in zip(scores, output_ids, strict=True):
                logprobs = F.log_softmax(score[0], dim=-1)
                logprob = logprobs[token].item()
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                    'logprob': logprob,
                })
                completions_logprob += logprob
            
            completions_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            if 'sparse' in self.mode:
                print(completions_text)
                print(self.detector.detect(completions_text))
                print(self.logit_processor_lst[0].last_input)
            return completions_text, completions_tokens