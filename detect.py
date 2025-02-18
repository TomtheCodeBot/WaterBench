from watermark import *
from tqdm import tqdm
from pred import load_model_and_tokenizer, seed_everything, str2bool
import argparse
import os
import json
import torch

def main(args):
    seed_everything(42)
    model2path = json.load(open("config/model2path.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model name
    model_name = args.input_dir.split("/")[-1].split("_")[0]
    # define your model
    tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, load_token_only=True)
    all_token_ids = list(tokenizer.get_vocab().values())
    vocab_size = len(all_token_ids)
    
    # get gamma and delta
    if "gpt" in args.input_dir:
        gamma = float(args.input_dir.split("_g")[2].split("_")[0])
    else:
        gamma = float(args.input_dir.split("_g")[1].split("_")[0])
    if "dipmark" not in args.input_dir:
        delta = float(args.input_dir.split("_d")[1].split("_")[0])
    # get all files from input_dir
    files = os.listdir(args.input_dir)
    # get all json files
    json_files = [f for f in files if f.endswith(".jsonl")]
    os.makedirs(args.input_dir + "/z_score", exist_ok=True)
    if args.mission != "all":
        json_files = [f for f in files if args.mission in f]
    counter = 0
    for json_file in json_files:
        print(f"{json_file} has began.........")
        # read jsons
        with open(os.path.join(args.input_dir, json_file), "r") as f:
            # lines
            lines = f.readlines()
            # texts
            prompts = [json.loads(line)["prompt"] for line in lines]
            print(len(lines))
            texts = [json.loads(line)["pred"] for line in lines]
            print(f"texts[0] is: {texts[0]}")
            if "new" in args.input_dir.split("/")[-1]:
                tokens = [json.loads(line)["completions_tokens"] for line in lines] 
            
            
        
        if "notagsparse" in args.input_dir:
            detector = NoTagSparseDetector(
                    tokenizer = tokenizer,
                    gamma=gamma,
                    delta=delta,
                    prompt_slice=None,
                    hard_encode=True if "hard" in args.input_dir else False,
                    allowed_pos_tag=None,
                    modular = delta
                    #seeding_scheme="selfhash"
                    )
        elif "old" in args.input_dir or "_no" in args.input_dir.split("/")[-1]:
            
            detector = OldWatermarkDetector(tokenizer=tokenizer,
                                            vocab=all_token_ids,
                                            gamma=gamma,
                                            delta=delta,
                                            dynamic_seed="markov_1",
                                            device=device)
        elif "dipmark" in args.input_dir:
            
            tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device,load_token_only=True)
            dipmark_config = DIPConfig(tokenizer=tokenizer,vocab=list(tokenizer.get_vocab().values()),
                               gamma =gamma,device = "cuda")
            dipmark_utils = DIPUtils(dipmark_config,state_indicator=1)
            detector = DIPDetector(config=dipmark_config,utils=dipmark_utils)
            print("RUNNING DIPMARK DETECTION")
        elif "ewd" in args.input_dir:
            model,tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
            detector = EWDWWatermarkDetector(vocab=all_token_ids,
                                            gamma=gamma,
                                            delta=delta,
                                            tokenizer=tokenizer,
                                            model=model,
                                            device=device)
            
        elif "sweet" in args.input_dir:
            print("RUNNING SWEET")
            model,tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
            detector = SweetDetector(vocab=all_token_ids,
                                            gamma=gamma,
                                            delta=delta,
                                            tokenizer=tokenizer,
                                            model=model)
        elif "og" in args.input_dir:
            if "ogv2" in args.input_dir:
                detector = OGWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=gamma,
                                                        delta=delta,
                                                        seeding_scheme = "selfhash",
                                                        device=device,
                                                        tokenizer=tokenizer,
                                                        z_threshold=args.threshold,)
            else:
                detector = OGWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=gamma,
                                                        delta=delta,
                                                        seeding_scheme = "lefthash",
                                                        device=device,
                                                        tokenizer=tokenizer,
                                                        z_threshold=args.threshold,
                                                        ignore_repeated_ngrams=False)
        
        elif "onebitsparsenormalhashshuffletag" in args.input_dir:
            detector = SparseOneBitNormalHashRandomTagDetector(
                    tokenizer = tokenizer,
                    gamma=gamma,
                    delta=delta,
                    prompt_slice=None,
                    hard_encode=True if "hard" in args.input_dir else False
                )
        elif 'sparsev2seedednormalhash'in args.input_dir :
            
            detector = SparseV2RandomNormalHashWatermarkDetector(
                tokenizer = tokenizer,
                secret_watermark= "password",
                prompt_slice=None,
                random_bit_string="random" in args.input_dir)
            
        elif "onebitsparsenormalhash" in args.input_dir:
            if "_onebitsparsenormalhash_" in args.input_dir:
                detector = SparseOneBitNormalHashDetector(
                    tokenizer = tokenizer,
                    gamma=gamma,
                    delta=delta,
                    prompt_slice=None,
                    hard_encode=True if "hard" in args.input_dir else False
                )
            else:
                pos_tags = args.input_dir.split("/")[-1].split("_")[1].split("onebitsparse")[1].split("-")
                
                pos_tags = list(filter(None, pos_tags))
                print(pos_tags)
                detector = SparseOneBitNormalHashDetector(
                    tokenizer = tokenizer,
                    gamma=gamma,
                    delta=delta,
                    prompt_slice=None,
                    hard_encode=True if "hard" in args.input_dir else False,
                    allowed_pos_tag=pos_tags
                )
        elif "onebitsparse" in args.input_dir:
            if "_onebitsparse_" in args.input_dir:
                detector = SparseOneBitDetector(
                    tokenizer = tokenizer,
                    gamma=gamma,
                    delta=delta,
                    prompt_slice=None,
                    hard_encode=True if "hard" in args.input_dir else False
                )
            else:
                pos_tags = args.input_dir.split("/")[-1].split("_")[1].split("onebitsparse")[1].split("-")
                
                pos_tags = list(filter(None, pos_tags))
                print(pos_tags)
                detector = SparseOneBitDetector(
                    tokenizer = tokenizer,
                    gamma=gamma,
                    delta=delta,
                    prompt_slice=None,
                    hard_encode=True if "hard" in args.input_dir else False,
                    allowed_pos_tag=pos_tags
                )
        elif "new" in args.input_dir:
            detector = NewWatermarkDetector(tokenizer=tokenizer,
                                        vocab=all_token_ids,
                                        gamma=gamma,
                                        delta=delta,
                                        dynamic_seed="markov_1",
                                        device=device,
                                        # vocabularys=vocabularys,
                                        )
        
        elif "sparsev2seeded" in args.input_dir:
            if "random"in args.input_dir:
                detector = SparseV2RandomWatermarkDetector(
                    tokenizer = tokenizer,
                    prompt_slice = None,
                    secret_watermark= "password",
                    random_bit_string=True)
            else:
                detector = SparseV2RandomWatermarkDetector(
                    tokenizer = tokenizer,
                    prompt_slice = None,
                    secret_watermark= "password")
            
        elif "sparsev2" in args.input_dir:
            if "random"in args.input_dir:
                detector = SparseV2WatermarkDetector(
                    tokenizer = tokenizer,
                    prompt_slice = None,
                    secret_watermark= "password",
                    random_bit_string=True)
            else:
                detector = SparseV2WatermarkDetector(
                    tokenizer = tokenizer,
                    prompt_slice = None,
                    secret_watermark= "password")
        elif "v2" in args.input_dir:
            detector = WatermarkDetector(
                vocab=all_token_ids,
                gamma=gamma,
                z_threshold=args.threshold,tokenizer=tokenizer,
                seeding_scheme=args.seeding_scheme,
                device=device,
                normalizers=args.normalizers,
                ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                select_green_tokens=args.select_green_tokens)
            
        elif "gpt" in args.input_dir:
            if "phi" in model_name:
                vocab_size = 32064
            detector = GPTWatermarkDetector(
                fraction=gamma,
                strength=delta,
                vocab_size=vocab_size,
                watermark_key=args.wm_key)
            
        
        elif "sparse" in args.input_dir:
            detector = StegoWatermarkDetector(
                tokenizer = tokenizer,
                secret_watermark= "password")
        
            
        z_score_list = []
        wm_pred = []
        for idx, cur_text in tqdm(enumerate(texts), total=len(texts)):
            #print("cur_text is:", cur_text)
            
            gen_tokens = tokenizer.encode(cur_text, return_tensors="pt", truncation=True, add_special_tokens=False)
            #print("gen_tokens is:", gen_tokens)
            prompt = prompts[idx]
            
            #input_prompt = tokenizer.encode(prompt, return_tensors="pt", truncation=True,add_special_tokens=False)
            input_prompt = tokenizer.encode("[INST][/INST]", return_tensors="pt", truncation=True,add_special_tokens=False)
            
            # if "v2" in args.input_dir and len(gen_tokens[0]) >= args.test_min_tokens:
            #     z_score_list.append(detector.detect(cur_text)["z_score"])
            
            # elif len(gen_tokens[0]) >= 1:
            #     if "old" in args.input_dir or "no" in args.input_dir:
            #         # print("gen_tokens is:", gen_tokens)
            #         z_score_list.append(detector.detect(tokenized_text=gen_tokens, inputs=input_prompt))
                
            #     elif "gpt" in args.input_dir:
            #         z_score_list.append(detector.detect(gen_tokens[0]))
            #     elif "new" in args.input_dir:
            #         z_score_list.append(detector.detect(tokenized_text=gen_tokens, tokens=tokens[idx], inputs=input_prompt))
                
            #     else:   
            #         print(f"Warning: sequence {idx} is too short to test. Which is ", gen_tokens[0])
            
            # else:
            #     print(f"Warning: sequence {idx} is too short to test. Which is ", gen_tokens[0])
            
            if len(gen_tokens[0]) >= args.test_min_tokens:
                if "onebit" in args.input_dir or "notags" in args.input_dir:
                    z_score_list.append(detector.detect(cur_text))
                elif "sparse" in args.input_dir:
                    wm_pred.append(detector.detect(cur_text))
                elif "v2" in args.input_dir:
                    z_score_list.append(detector.detect(cur_text)["z_score"])
                elif "dipmark" in args.input_dir:
                    z_score_list.append(detector.detect(cur_text)["z_score"])
                elif "og" in args.input_dir:
                    z_score_list.append(detector.detect(cur_text)["z_score"])
                    print(z_score_list[-1])
                elif "ewd" in args.input_dir:
                    print("gen_tokens is:", gen_tokens)
                    print(input_prompt)
                    full_text = torch.cat((input_prompt, gen_tokens), -1)
                    z_score_list.append(detector.detect(tokenized_text=full_text[0], tokenized_prefix=input_prompt[0])["z_score"])
                elif "sweet" in args.input_dir:
                    print("gen_tokens is:", gen_tokens)
                    print(input_prompt)
                    print("SWEET detection")
                    full_text = torch.cat((input_prompt, gen_tokens), -1)
                    z_score_list.append(detector.detect(tokenized_text=full_text[0], tokenized_prefix=input_prompt[0])["z_score"])
                
                elif "old" in args.input_dir or "no" in args.input_dir:
                    print("gen_tokens is:", gen_tokens)
                    z_score_list.append(detector.detect(tokenized_text=gen_tokens, inputs=input_prompt))
                
                elif "gpt" in args.input_dir:
                      z_score_list.append(detector.detect(gen_tokens[0]))
                elif "new" in args.input_dir:
                      z_score_list.append(detector.detect(tokenized_text=gen_tokens, tokens=tokens[idx], inputs=input_prompt))
                
            else:   
                print(f"Warning: sequence {idx} is too short to test.")
                    
            # if len(gen_tokens[0]) >= 1:
            #     if "old" in args.input_dir or "no" in args.input_dir:
            #         print("gen_tokens is:", gen_tokens)
            #         z_score_list.append(detector.detect(tokenized_text=gen_tokens, inputs=input_prompt))
                
            #     elif "gpt" in args.input_dir:
            #         z_score_list.append(detector.detect(gen_tokens[0]))
            #     elif "new" in args.input_dir:
            #         z_score_list.append(detector.detect(tokenized_text=gen_tokens, tokens=tokens[idx], inputs=input_prompt))
            # else:   
            #     print(f"Warning: sequence {idx} is too short to test.")
        if "sparse" not in args.input_dir or "onebit" in args.input_dir or"notag" in args.input_dir:
            save_dict = {
                'z_score_list': z_score_list,
                'avarage_z': torch.mean(torch.tensor(z_score_list)).item(),
                'wm_pred': [1 if z > args.threshold else 0 for z in z_score_list]
                }
        else:
            save_dict = {
                'wm_pred': [1 if x else 0 for x in wm_pred]
                }
        wm_pred_average = torch.mean(torch.tensor(save_dict['wm_pred'], dtype=torch.float))
        save_dict.update({'wm_pred_average': wm_pred_average.item()})   
        
        print(save_dict)
        # average_z = torch.mean(z_score_list)
        z_file = json_file.replace('.jsonl', f'_{gamma}_{delta}_{args.threshold}_z.jsonl')
        output_path = os.path.join(args.input_dir + "/z_score", z_file)
        with open(output_path, 'w') as fout:
            json.dump(save_dict, fout)
        if "onebit" in args.input_dir:
            counter += detector.all_observed
        
    print(counter)
        


parser = argparse.ArgumentParser(description="Process watermark to calculate z-score for every method")

parser.add_argument(
    "--input_dir",
    type=str,
    default="/data2/tsq/WaterBench/pred/llama2-7b-chat-4k_old_g0.5_d5.0")
parser.add_argument( # for gpt watermark
        "--wm_key", 
        type=int, 
        default=0)

parser.add_argument(
    "--threshold",
    type=float,
    default=6.0)

parser.add_argument(
    "--test_min_tokens",
    type=int, 
    default=2)

parser.add_argument( # for v2 watermark
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )

parser.add_argument( # for v2 watermark
    "--normalizers",
    type=str,
    default="",
    help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
)

parser.add_argument( # for v2 watermark
    "--ignore_repeated_bigrams",
    type=str2bool,
    default=False,
    help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
)

parser.add_argument( # for v2 watermark
    "--select_green_tokens",
    type=str2bool,
    default=True,
    help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
)

parser.add_argument( 
    "--mission",
    type=str,
    default="all",
    help="mission-name",
)
args = parser.parse_args()

main(args)

