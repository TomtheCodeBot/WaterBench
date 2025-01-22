import sklearn.metrics as metrics
import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
def get_roc(human_scores, machine_scores, max_fpr=1.0):
    fpr, tpr, _ = roc_curve([0] * len(human_scores) + [1] * len(machine_scores), human_scores + machine_scores)
    fpr_auc = [x for x in fpr if x <= max_fpr]
    tpr_auc = tpr[:len(fpr_auc)]
    roc_auc = auc(fpr_auc, tpr_auc)
    return fpr.tolist(), tpr.tolist(), float(roc_auc), float(roc_auc) * (1.0 / max_fpr)
UNIVERSAL_POS_TAG = {
    ".": ["."],
    "CONJ": ["CC"],
    "NUM": ["CD"],
    "X": ["CD|RB","FW","LS","RN","SYM","UH","WH"],
    "DET": ["DT","EX","PDT","WDT"],
    "ADP": ["IN","IN|RP"],
    "ADJ": ["JJ","JJR","JJRJR","JJS","JJ|RB","JJ|VBG"],
    "VERB": ["MD","VB","VBD","VBD|VBN","VBG","VBG|NN","VBN","VBP","VBP|TO","VBZ","VP"],
    "NOUN": ["NN","NNP","NNPS","NNS","NN|NNS","NN|SYM","NN|VBG","NP"],
    "PRT": ["POS","PRT","RP","TO"],
    "PRON": ["PRP","PRP$","PRP|VBP","WP","WP$"],
    "ADV": ["RB","RBR","RBS","RB|RP","RB|VBG","WRB"]
}
def bulk_auc_calc(main_directory, threshold="4.0"):
    # Define the column names
    columns = ['model', 'finance_qa', 'longform_qa', 'qmsum', 'multi_news',"all"]
    sepcified_dataset_names  = [ 'finance_qa', 'longform_qa', 'qmsum', 'multi_news']
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)
    if isinstance(main_directory, str):
        with os.scandir(main_directory) as entries:
            main_directory = [f"{main_directory}/{entry.name}" for entry in entries if entry.is_dir()]
    all_watermark_zscore = []
    all_baseline_zscore = []
    for directory in main_directory:

        try:
            mode = "_".join(directory.split("/")[-1].split("_")[1:])
            watermark_method = mode.split("_")[0]
            gamma = mode.split("_g")[-1].split("_")[0]
            delta = mode.split("_d")[-1].split("_")[0]
            hard_mode = mode.split("_")[-1] == "hard"
            model = directory.split("/")[-1].split("_")[0]
            data_to_add = {"model": mode}
            if hard_mode:
                base_path = f"detect_human/{model}/ref_{watermark_method}_hard_g{gamma}_d{delta}/human_generation_z"
            else:
                base_path = f"detect_human/{model}/ref_{watermark_method}_g{gamma}_d{delta}/human_generation_z"
            watermarked_path = directory + "/z_score"
            files = os.listdir(directory)
            # get all json files
            dataset_names = [f.split(".")[0] for f in files if f.endswith(".jsonl")]
            for dataset_name in dataset_names:
                if dataset_name not in sepcified_dataset_names:
                    continue
                with open(os.path.join(watermarked_path, f"{dataset_name}_{gamma}_{delta}_{threshold}_z.jsonl"), "r") as f:
                    watermark_z_scores = json.loads(f.readlines()[0])["z_score_list"]
                    all_watermark_zscore.extend(watermark_z_scores)
                with open(os.path.join(base_path, f"{dataset_name}_{threshold}_z.jsonl"), "r") as f:
                    baseline_z_scores = json.loads(f.readlines()[0])["z_score_list"]
                    all_baseline_zscore.extend(baseline_z_scores)
                data_to_add[dataset_name] = get_roc(baseline_z_scores, watermark_z_scores)
            data_to_add["all"] = get_roc(all_baseline_zscore, all_watermark_zscore)
            df = df._append(data_to_add, ignore_index=True)
        except FileNotFoundError as e:
            print(e)
    return df
def consolidate_results(path_list, threshold="4.0"):
    consolidated_df = pd.DataFrame()
    for path in path_list:
        df = bulk_auc_calc(path, threshold)
        consolidated_df = pd.concat([consolidated_df, df], ignore_index=True)
    consolidated_df.to_csv("ROC_AUC.CSV",index=False)
    return consolidated_df
# Example usage
path_list = ["/cluster/tufts/laolab/kdoan02/selected_results"]
consolidated_df = consolidate_results(path_list)
print(consolidated_df)