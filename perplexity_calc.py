import os
import json
import lmppl
from tqdm import tqdm
from evaluate import load
import json
model2path = json.load(open("config/model2path.json", "r"))

def calculate_perplexity_for_file(model_name, file_path,key):
    scorer = lmppl.LM(model2path[model_name])
    perplexities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'Processing {os.path.basename(file_path)}'):
            data = json.loads(line)
            if key == "answers":
                text = data.get(key, '')[0]
            else:
                text = data.get(key, '')  # Assuming the text is under the 'text' key
            perplexities.append(scorer.get_perplexity(text))
            print(text)
            print(perplexities[-1])
    
    if perplexities:
        average_perplexity = sum(perplexities) / len(perplexities)
    else:
        average_perplexity = float('inf')
    
    return average_perplexity

def calculate_perplexity_for_file_hf(model_name, file_path,key):
    perplexity = load("perplexity", module_type="metric")
    

    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'Processing {os.path.basename(file_path)}'):
            data = json.loads(line)
            if key == "answers":
                text = data.get(key, '')[0]
            else:
                text = data.get(key, '')  # Assuming the text is under the 'text' key
            texts.append(text)
    
    results = perplexity.compute(predictions=texts, model_id=model2path[model_name],batch_size=1)
    
    return results

def main(directory, model_name,key,mode):
    jsonl_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    output_dir = os.path.join(directory, 'perplexity')
    os.makedirs(output_dir, exist_ok=True)
    
    for jsonl_file in jsonl_files:
        file_path = os.path.join(directory, jsonl_file)
        if mode=="hf":
            print("RUNNING HF")
            average_perplexity = calculate_perplexity_for_file_hf(model_name, file_path,key)
            output_file = os.path.join(output_dir, f'{os.path.splitext(jsonl_file)[0]}_perplexity_hf.json')
        else:
            average_perplexity = calculate_perplexity_for_file(model_name, file_path,key)
            output_file = os.path.join(output_dir, f'{os.path.splitext(jsonl_file)[0]}_perplexity_lmppl.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(average_perplexity, f, ensure_ascii=False)
        
        print(f'Perplexity for {jsonl_file} saved to {output_file}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate average perplexity for JSONL files in a directory.")
    parser.add_argument('directory', type=str, help='Directory containing JSONL files')
    parser.add_argument('model_name', type=str, help='Model name to be used for calculating perplexity')
    parser.add_argument('key', type=str, help='Key to the text for calculating perplexity')
    parser.add_argument('mode',default=None, type=str, help='Select version of perplexity calculation to run')
    args = parser.parse_args()
    main(args.directory, args.model_name,args.key,args.mode)
