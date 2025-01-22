from watermark import *
from tqdm import tqdm
from pred import load_model_and_tokenizer, seed_everything, str2bool
import argparse
import os
import json
import torch
import re

def main(args):
    seed_everything(42)
    model2path = json.load(open("config/model2path.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model name
    model_name = args.reference_dir.split("/")[-1].split("_")[0]
    print(args.reference_dir)
    # define your model
    tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, load_token_only=True)
    all_token_ids = list(tokenizer.get_vocab().values())
    vocab_size = len(all_token_ids)
    if "phi" in model_name:
        vocab_size = 32064
    
    
    all_input_dir = "./pred/"
    # get gamma and delta
    pattern_dir = r"(?P<model_name>.+)_(?P<mode>old|v2|gpt|new|no|ewd|notagsparse|sparse|sparsev2|sweet|ogv2|onebitsparsenormalhashshuffletag|onebitsparse|onebitsparsenormalhash|)_g(?P<gamma>.+)_d(?P<delta>\d+(\.\d+)?)"
    if "onebitsparse" in args.reference_dir or "notagsparse" in args.reference_dir:
        param_section = args.reference_dir.split("/")[-1].split("_")
        gamma_ref = float(param_section[2][1:])
        delta_ref = float(param_section[3][1:])
        print(gamma_ref,delta_ref)
        mode_ref = param_section[1]
    else:
        
        
        pattern_mis = r"(?P<misson_name>[a-zA-Z_]+)_(?P<gamma>\d+(\.\d+)?)_(?P<delta>.+)_z"
        
        matcher_ref = re.match(pattern_dir, args.reference_dir)

        mode_ref = matcher_ref.group("mode")

        gamma_ref = float(matcher_ref.group("gamma"))
        delta_ref = float(matcher_ref.group("delta"))
    bl_type_ref = "None"
    bl_type_ref = (args.reference_dir.split("_")[-1]).split(".")[0]
    
    if bl_type_ref != "hard":
        if "old" in args.reference_dir:
            bl_type_ref = "soft"
            mode_ref += "_" + bl_type_ref
        else:
            bl_type_ref = "None"   
    else:
        mode_ref += "_" + bl_type_ref
    
    print("mode_ref is:", mode_ref)  
    
    if args.detect_dir != "human_generation":
        matcher_det = re.match(pattern_dir, args.detect_dir)
        mode_det = matcher_det.group("mode")
        gamma_det = float(matcher_det.group("gamma"))
        delta_det = float(matcher_det.group("delta"))
        bl_type_det = "None"
        bl_type_det = (args.detect_dir.split("_")[-1]).split(".")[0]


        if bl_type_det != "hard":
            if "old" in args.detect_dir:
                bl_type_det = "soft"
                mode_det += "_" + bl_type_det
            else:
                bl_type_det = "None"
        else:
            mode_det += "_" + bl_type_det

        # print("bl_type_det is:", bl_type_det)  
        # print("mode_det is:", mode_det)    
    # get all files from detect_dir
    
    files = os.listdir(all_input_dir + args.detect_dir)
    
    # get all json files
    json_files = [f for f in files if f.endswith(".jsonl")]
    
    ref_dir = f"./detect_human/{model_name}/ref_{mode_ref}_g{gamma_ref}_d{delta_ref}"
    
    os.makedirs(f"./detect_human/{model_name}", exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    if args.detect_dir == "human_generation":
          os.makedirs(ref_dir + "/human_generation_z", exist_ok=True)
    else:
        os.makedirs(ref_dir + f"/{mode_det}_g{gamma_det}_d{delta_det}_z", exist_ok=True)
    
    if "notagsparse" in args.reference_dir:
        detector = NoTagSparseDetector(
                tokenizer = tokenizer,
                gamma=gamma_ref,
                delta=delta_ref,
                prompt_slice=None,
                hard_encode=True if "hard" in args.reference_dir else False,
                allowed_pos_tag=None,
                modular = delta_ref
                #seeding_scheme="selfhash"
                )
    elif "old" in args.reference_dir or "_no" in args.reference_dir.split("/")[-1]:
            detector = OldWatermarkDetector(tokenizer=tokenizer,
                                            vocab=all_token_ids,
                                            gamma=gamma_ref,
                                            delta=delta_ref,
                                            dynamic_seed="markov_1",
                                            device=device)
    elif "og" in args.reference_dir:
        if "ogv2" in args.reference_dir:
            detector = OGWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=gamma_ref,
                                                    delta=delta_ref,
                                                    seeding_scheme = "selfhash",
                                                    device=device,
                                                    tokenizer=tokenizer,
                                                    z_threshold=args.threshold,)
        else:
            detector = OGWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=gamma_ref,
                                                    delta=delta_ref,
                                                    seeding_scheme = "lefthash",
                                                    device=device,
                                                    tokenizer=tokenizer,
                                                    z_threshold=args.threshold,
                                                    ignore_repeated_ngrams=False)
    elif "ewd" in args.reference_dir:
        model,tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
        detector = EWDWWatermarkDetector(vocab=all_token_ids,
                                        gamma=gamma_ref,
                                        delta=delta_ref,
                                        tokenizer=tokenizer,
                                        model=model,
                                        device=device)
    elif "sweet" in args.reference_dir:
        model,tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
        detector = SweetDetector(vocab=all_token_ids,
                                            gamma=gamma_ref,
                                            delta=delta_ref,
                                            tokenizer=tokenizer,
                                            model=model)
    
    elif "onebitsparsenormalhashshuffletag" in args.reference_dir:
            detector = SparseOneBitNormalHashRandomTagDetector(
                    tokenizer = tokenizer,
                    gamma=gamma_ref,
                    delta=delta_ref,
                    prompt_slice=None,
                    hard_encode=True if "hard" in args.reference_dir else False
                )
    elif "onebitsparsenormalhash" in args.reference_dir:
        if "_onebitsparsenormalhash_" in args.reference_dir:
            detector = SparseOneBitNormalHashDetector(
                tokenizer = tokenizer,
                gamma=gamma_ref,
                delta=delta_ref,
                prompt_slice=None,
                hard_encode=True if "hard" in args.reference_dir else False
            )
        else:
            pos_tags = args.reference_dir.split("/")[-1].split("_")[1].split("onebitsparse")[1].split("-")
            
            pos_tags = list(filter(None, pos_tags))
            print("onebitsparsenormalhash", pos_tags)
            detector = SparseOneBitNormalHashDetector(
                tokenizer = tokenizer,
                gamma=gamma_ref,
                delta=delta_ref,
                prompt_slice=None,
                hard_encode=True if "hard" in args.reference_dir else False,
                allowed_pos_tag=pos_tags
            )
    elif "onebitsparse" in args.reference_dir:
        if "_onebitsparse_" in args.reference_dir:
                detector = SparseOneBitDetector(
                    tokenizer = tokenizer,
                    gamma=gamma_ref,
                    delta=delta_ref,
                    prompt_slice=None,
                    hard_encode=True if "hard" in args.reference_dir else False
                )
        else:
            pos_tags = args.reference_dir.split("/")[-1].split("_")[1].split("onebitsparse")[1].split("-")
            pos_tags = list(filter(None, pos_tags))
            print(pos_tags,gamma_ref,delta_ref)
            detector = SparseOneBitDetector(
                tokenizer = tokenizer,
                gamma=gamma_ref,
                delta=delta_ref,
                prompt_slice=None,
                hard_encode=True if "hard" in args.reference_dir else False,
                allowed_pos_tag=pos_tags
            )
    elif "new" in args.reference_dir:
        detector = NewWatermarkDetector(tokenizer=tokenizer,
                                    vocab=all_token_ids,
                                    gamma=gamma_ref,
                                    delta=delta_ref,
                                    dynamic_seed="markov_1",
                                    device=device,
                                    # vocabularys=vocabularys,
                                    )
    elif "sparsev2" in args.reference_dir:
        if "random"in args.reference_dir:
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
    elif "v2" in args.reference_dir:
        detector = WatermarkDetector(
            vocab=all_token_ids,
            gamma=gamma_ref,
            z_threshold=args.threshold,tokenizer=tokenizer,
            seeding_scheme=args.seeding_scheme,
            device=device,
            normalizers=args.normalizers,
            ignore_repeated_bigrams=args.ignore_repeated_bigrams,
            select_green_tokens=args.select_green_tokens)
        
    elif "gpt" in args.reference_dir:
        detector = GPTWatermarkDetector(
            fraction=gamma_ref,
            strength=delta_ref,
            vocab_size=vocab_size,
            watermark_key=args.wm_key)
    
    elif "sparse" in args.reference_dir:
        detector = StegoWatermarkDetector(
            tokenizer = tokenizer,
            secret_watermark= "password")
        
    prompts = []        
    for json_file in json_files:
        print(f"{json_file} has began.........")
        #if args.detect_dir == "human_generation":
        #    with open(os.path.join(all_input_dir + args.reference_dir, json_file), "r") as f:
        #        lines = f.readlines()
        #        
        #        prompts = [json.loads(line)["prompt"] for line in lines]
        #        print("len of prompts is", len(prompts))
        # read jsons
        with open(os.path.join(all_input_dir + args.detect_dir, json_file), "r") as f:
            # lines
            lines = f.readlines()
            # texts
            if args.detect_dir != "human_generation":
                prompts = [json.loads(line)["prompt"] for line in lines]
                texts = [json.loads(line)["pred"] for line in lines]
            else:
                prompts = [json.loads(line)["prompt"] for line in lines]
                texts = [json.loads(line)["answers"][0] for line in lines]
                
            print(f"texts[0] is: {texts[0]}")
            tokens = [json.loads(line)["completions_tokens"] for line in lines]
                
        z_score_list = []
        wm_pred = []
        
        for idx, cur_text in tqdm(enumerate(texts), total=len(texts)):
            #print("cur_text is:", cur_text)
            
            gen_tokens = tokenizer.encode(cur_text, return_tensors="pt", truncation=True, add_special_tokens=False)
            #print("gen_tokens is:", gen_tokens)
            prompt = prompts[idx]
            
            #input_prompt = tokenizer.encode(prompt, return_tensors="pt", truncation=True,add_special_tokens=False)
            input_prompt = tokenizer.encode("[INST][/INST]", return_tensors="pt", truncation=True,add_special_tokens=False)
            print(input_prompt)
            
            if len(gen_tokens[0]) >= args.test_min_tokens:
                
                if "v2" in args.reference_dir and not "sparse" in args.reference_dir and not "og" in args.reference_dir :
                    z_score_list.append(detector.detect(cur_text)["z_score"])
                        
            if len(gen_tokens[0]) >= 1:
                if "onebit" in args.reference_dir:
                    z_score_list.append(detector.detect(cur_text))
                elif "gpt" in args.reference_dir:
                    z_score_list.append(detector.detect(gen_tokens[0]))
                elif "ewd" in args.reference_dir:
                    print("gen_tokens is:", gen_tokens)
                    print(input_prompt)
                    full_text = torch.cat((input_prompt, gen_tokens), -1)
                    z_score_list.append(detector.detect(tokenized_text=full_text[0], tokenized_prefix=input_prompt[0])["z_score"])
                elif "sweet" in args.reference_dir:
                    print("gen_tokens is:", gen_tokens)
                    print(input_prompt)
                    full_text = torch.cat((input_prompt, gen_tokens), -1)
                    z_score_list.append(detector.detect(tokenized_text=full_text[0], tokenized_prefix=input_prompt[0])["z_score"])
                elif "sparse" in args.reference_dir:
                    wm_pred.append(detector.detect(cur_text))
                elif "og" in args.reference_dir:
                    z_score_list.append(detector.detect(cur_text)["z_score"])
                elif "old" in args.reference_dir or "no" in args.reference_dir:
                    z_score_list.append(detector.detect(tokenized_text=gen_tokens, inputs=input_prompt))
                elif "new" in args.reference_dir:
                    z_score_list.append(detector.detect(tokenized_text=gen_tokens, tokens=tokens[idx], inputs=input_prompt))
                
            else:        
                print(f"Warning: sequence {idx} is too short to test.")
        if "sparse" not in args.reference_dir or "onebit" in args.reference_dir or "notagsparse" in args.reference_dir:
            print(z_score_list)
            save_dict = {
                'z_score_list': z_score_list,
                'avarage_z': torch.mean(torch.tensor(z_score_list,dtype=torch.float)).item(),
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
        z_file = json_file.replace('.jsonl', f'_{args.threshold}_z.jsonl')
        
        if args.detect_dir != "human_generation":
            if "random"in args.reference_dir:
                output_path = os.path.join(ref_dir + f"/{mode_det}_g{gamma_det}_d{delta_det}_z_random", z_file)
            else:
                output_path = os.path.join(ref_dir + f"/{mode_det}_g{gamma_det}_d{delta_det}_z", z_file)
        
        else:
            output_path = os.path.join(ref_dir + "/human_generation_z", z_file)
        with open(output_path, 'w') as fout:
            json.dump(save_dict, fout)
    if "onebitsparse-" in args.reference_dir:
        print(detector.all_observed)
        


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
    default=4.0)

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
    "--reference_dir",
    type=str,
    default="/data2/tsq/WaterBench/pred/llama2-7b-chat-4k_v2_g0.25_d15.0",
    help="Which type as reference to test TN or FP",
)

parser.add_argument(
    "--detect_dir",
    type=str,
    default="/data2/tsq/WaterBench/pred/llama2-7b-chat-4k_v2_g0.25_d15.0",
    help="Which type need to be detected",
)
args = parser.parse_args()

main(args)

