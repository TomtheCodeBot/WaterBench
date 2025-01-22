import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import pandas as pd
import ast

from copy import deepcopy
# Given dictionary with multiple TPR and FPR lists
def literal_eval_converter(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


df = pd.read_csv('ROC_AUC.CSV', converters={col: literal_eval_converter for col in range(len(pd.read_csv('ROC_AUC.CSV', nrows=1).columns))})
name_dict=  {"ogv2":"SelfHash","gpt":"Unigram","old":"Hard","v2":"LeftHash","no":"No Watermark"}

roc_data = {}
for i in range(len(df)):
    new_dict = {}
    new_dict["fpr"] = df.iloc[i]["all"][0]
    print(new_dict)
    name = df.iloc[i]["model"].split("_")[0]
    if "NN" in name:
        name = "Noun"
    elif "VP" in name:
        name = "Verb"
    elif "DT" in name:
        name = "Determiner"
    else:
        name = name_dict[name]

    new_dict["tpr"] = df.iloc[i]["all"][1]
    roc_data[name] = deepcopy(new_dict)
# Plot ROC curve for each model in the dictionary


f = plt.figure()

colors =['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
styles = [
    {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
    {'color': 'red', 'linestyle': '--', 'marker': 's'},
    {'color': 'green', 'linestyle': '-.', 'marker': 'D'},
    {'color': 'purple', 'linestyle': ':', 'marker': '^'},
    {'color': 'orange', 'linestyle': '-', 'marker': 'v'},
    {'color': 'brown', 'linestyle': '--', 'marker': '<'},
    {'color': 'pink', 'linestyle': '-.', 'marker': '>'}
]
for i, (model, data) in enumerate(roc_data.items()):
    fpr = data['fpr']
    tpr = data['tpr']
    roc_auc = auc(fpr, tpr)
    style = styles[i]
    plt.plot(fpr, tpr, color=style["color"],linestyle=style["linestyle"], lw=2,alpha=0.5, label=f'{model} (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
f.savefig("auc_curve.pdf", bbox_inches='tight')