




# without haversine
# import os
# import yaml
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from transformers import BertTokenizerFast, BertConfig, BertPreTrainedModel, BertModel
# from tqdm.auto import tqdm
# from sklearn.metrics import accuracy_score, precision_recall_curve
# from Levenshtein import ratio

# class BertJointHead(BertPreTrainedModel):
#     """
#     Joint binary match head + dual entity-ID heads (edit_dist input)
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         hidden = config.hidden_size
#         self.match_head = torch.nn.Linear(hidden + 1, config.num_labels)
#         self.idA_head  = torch.nn.Linear(hidden, config.num_labels_entity)
#         self.idB_head  = torch.nn.Linear(hidden, config.num_labels_entity)
#         self.post_init()

#     def forward(self,
#                 input_ids,
#                 attention_mask=None,
#                 token_type_ids=None,
#                 edit_dist=None,
#                 **kwargs):
#         out = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True
#         )
#         pooled = out.pooler_output
#         ed = edit_dist.unsqueeze(1).to(pooled.dtype)
#         feats = torch.cat([pooled, ed], dim=1)
#         logits_match = self.match_head(feats)
#         return logits_match

# if __name__ == '__main__':
#     # Load configuration
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
#     inf_cfg = cfg.get('inference', {})

#     # Locate trained model
#     model_dir = os.path.abspath(cfg['training']['output_dir'])
#     if not os.path.isdir(model_dir):
#         raise FileNotFoundError(f"Model directory not found: {model_dir}")

#     # Device setup
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Determine num_labels_entity from training CSV
#     train_csv = os.path.join(base_dir, cfg['data']['input_csv'])
#     df_train = pd.read_csv(train_csv)
#     num_entities = int(df_train[['clusterA','clusterB']].max().max()) + 1

#     # Load tokenizer and model config
#     bert_conf = BertConfig.from_pretrained(
#         model_dir,
#         num_labels=int(cfg['model']['num_labels'])
#     )
#     bert_conf.num_labels_entity = int(num_entities)

#     # Load model + tokenizer
#     model = BertJointHead.from_pretrained(model_dir, config=bert_conf)
#     model.to(device).eval()
#     tokenizer = BertTokenizerFast.from_pretrained(model_dir)
#     print("Loaded JointBERT model & tokenizer")

#     # Load inference data
#     addr_path  = os.path.join(base_dir, cfg['data']['test_csv'])
#     label_path = os.path.join(base_dir, cfg['data']['inference_labels'])
#     inf_df  = pd.read_csv(addr_path)
#     labels = pd.read_csv(label_path)
#     dup_set = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in labels.itertuples())

#     # Parameters
#     max_len   = int(cfg['model']['max_length'])
#     batch_sz  = int(inf_cfg.get('batch_size', 32))
#     max_pairs = inf_cfg.get('max_pairs', float('inf'))

#     print(f"Testing without Haversine filter, batch size = {batch_sz}")

#     probs, trues = [], []
#     buffer, count = [], 0
#     wrong_records = []  # collect wrong predictions

#     def process_batch(batch):
#         texts_a = [x['a'] for x in batch]
#         texts_b = [x['b'] for x in batch]
#         edits    = [ratio(a, b) for a, b in zip(texts_a, texts_b)]
#         enc = tokenizer(texts_a, texts_b,
#                         return_tensors='pt', padding='max_length', truncation=True,
#                         max_length=max_len)
#         enc = {k:v.to(device) for k,v in enc.items()}
#         edit_t = torch.tensor(edits, device=device)
#         with torch.no_grad():
#             logits = model(
#                 input_ids=enc['input_ids'],
#                 attention_mask=enc['attention_mask'],
#                 token_type_ids=enc['token_type_ids'],
#                 edit_dist=edit_t
#             )
#             scores = F.softmax(logits, dim=1)[:,1].cpu().tolist()
#         for item, p in zip(batch, scores):
#             pair = frozenset((item['i'], item['j']))
#             true = 1 if pair in dup_set else 0
#             pred = 1 if p >= float(cfg['inference'].get('threshold', 0.73)) else 0
#             probs.append(p)
#             trues.append(true)
#             if pred != true:
#                 wrong_records.append({
#                     'indexA': item['i'],
#                     'indexB': item['j'],
#                     'text_a': item['a'],
#                     'text_b': item['b'],
#                     'true_label': true,
#                     'pred_label': pred,
#                     'probability': p
#                 })

#     # Generate candidate pairs (no spatial filtering)
#     print("Starting exhaustive pairwise matching...")
#     for i, row_a in tqdm(inf_df.iterrows(), total=len(inf_df)):
#         for j in range(i+1, len(inf_df)):
#             if count >= max_pairs:
#                 print(f"1. count: {count} >= max_pair: {max_pairs}")
#                 break
#             buffer.append({'a': row_a['text'], 'b': inf_df.iloc[j]['text'],
#                            'i': row_a['orig_index'], 'j': inf_df.iloc[j]['orig_index']})
#             count += 1
#             if len(buffer) >= batch_sz:
#                 process_batch(buffer)
#                 buffer.clear()
#         if count >= max_pairs:
#             print(f"2. count: {count} >= max_pair: {max_pairs}")
#             break

#     if buffer:
#         process_batch(buffer)

#     print(f"Processed {count} pairs, found {sum(trues)} positives")

#     # Compute metrics and threshold sweep
#     prec, rec, ths = precision_recall_curve(trues, probs)
#     f1s = 2 * prec * rec / (prec + rec + 1e-8)
#     best = f1s.argmax()
#     best_t = ths[best]

#     print("\n=== RESULTS ===")
#     print(f"Best threshold: {best_t:.3f}")
#     print(f"Precision: {prec[best]:.4f}, Recall: {rec[best]:.4f}, F1: {f1s[best]:.4f}")
#     acc = accuracy_score(trues, [1 if p>=best_t else 0 for p in probs])
#     print(f"Accuracy: {acc:.4f}")

#     # Save outputs
#     out_dir = os.path.join(base_dir, cfg['training']['output_dir'])
#     os.makedirs(out_dir, exist_ok=True)
#     pd.DataFrame({'threshold': ths,
#                   'precision': prec[:-1],
#                   'recall': rec[:-1],
#                   'f1': f1s[:-1]
#                   }).to_csv(os.path.join(out_dir, 'threshold_sweep_no_hav.csv'), index=False)
#     if wrong_records:
#         pd.DataFrame(wrong_records).to_csv(os.path.join(out_dir, 'wrong_predictions_no_hav.csv'), index=False)
#         print(f"Wrong predictions saved to: {os.path.join(out_dir, 'wrong_predictions_no_hav.csv')}")
#     else:
#         print("No wrong predictions to save.")


import os
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from math import radians, sin, cos, sqrt, atan2
from transformers import BertTokenizerFast, BertConfig, BertPreTrainedModel, BertModel
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
from Levenshtein import ratio
import numpy as np

# Haversine distance for geospatial filtering
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

class BertJointHead(BertPreTrainedModel):
    """
    Joint binary match head + dual entity-ID heads (edit_dist input)
    """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        hidden = config.hidden_size
        self.match_head = torch.nn.Linear(hidden + 1, config.num_labels)
        self.idA_head  = torch.nn.Linear(hidden, config.num_labels_entity)
        self.idB_head  = torch.nn.Linear(hidden, config.num_labels_entity)
        self.post_init()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                edit_dist=None,
                **kwargs):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled = out.pooler_output
        ed = edit_dist.unsqueeze(1).to(pooled.dtype)
        feats = torch.cat([pooled, ed], dim=1)
        logits_match = self.match_head(feats)
        return logits_match

if __name__ == '__main__':
    # Load configuration
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
    inf_cfg = cfg.get('inference', {})

    # Locate trained model
    model_dir = os.path.abspath(cfg['training']['output_dir'])
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine num_labels_entity from training CSV
    train_csv = os.path.join(base_dir, cfg['data']['input_csv'])
    df_train = pd.read_csv(train_csv)
    num_entities = int(df_train[['clusterA','clusterB']].max().max()) + 1

    # Load tokenizer and model config
    bert_conf = BertConfig.from_pretrained(
        model_dir,
        num_labels=int(cfg['model']['num_labels'])
    )
    bert_conf.num_labels_entity = int(num_entities)

    # Load model + tokenizer
    model = BertJointHead.from_pretrained(model_dir, config=bert_conf)
    model.to(device).eval()
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    print("Loaded JointBERT model & tokenizer")

    # Load inference data
    addr_path  = os.path.join(base_dir, cfg['data']['test_csv'])
    label_path = os.path.join(base_dir, cfg['data']['inference_labels'])
    inf_df  = pd.read_csv(addr_path)
    labels = pd.read_csv(label_path)
    dup_set = set(frozenset((r.orig_indexA, r.orig_indexB)) for r in labels.itertuples())

    # Parameters
    thresh    = float(cfg['data']['distance_threshold_m'])
    max_len   = int(cfg['model']['max_length'])
    batch_sz  = int(inf_cfg.get('batch_size', 32))
    max_pairs = inf_cfg.get('max_pairs', float('inf'))

    print(f"Distance threshold = {thresh}m, batch size = {batch_sz}")

    probs, trues = [], []
    buffer, count = [], 0
    wrong_records = []  # collect wrong predictions

    def process_batch(batch):
        texts_a = [x['a'] for x in batch]
        texts_b = [x['b'] for x in batch]
        edits    = [ratio(a, b) for a, b in zip(texts_a, texts_b)]
        enc = tokenizer(texts_a, texts_b,
                        return_tensors='pt', padding='max_length', truncation=True,
                        max_length=max_len)
        enc = {k:v.to(device) for k,v in enc.items()}
        edit_t = torch.tensor(edits, device=device)
        with torch.no_grad():
            logits = model(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                token_type_ids=enc['token_type_ids'],
                edit_dist=edit_t
            )
            scores = F.softmax(logits, dim=1)[:,1].cpu().tolist()
        for item, p in zip(batch, scores):
            pair = frozenset((item['i'], item['j']))
            true = 1 if pair in dup_set else 0
            pred = 1 if p >= inf_cfg.get('threshold', 0.5) else 0
            probs.append(p)
            trues.append(true)
            # record wrong predictions with full context
            if pred != true:
                wrong_records.append({
                    'indexA': item['i'],
                    'indexB': item['j'],
                    'text_a': item['a'],
                    'text_b': item['b'],
                    'true_label': true,
                    'pred_label': pred,
                    'probability': p
                })

    # Generate candidate pairs
    print("Starting pairwise matching...")
    for i, row_a in tqdm(inf_df.iterrows(), total=len(inf_df)):
        for j in range(i+1, len(inf_df)):
            if count >= max_pairs:
                break
            row_b = inf_df.iloc[j]
            dist = haversine(row_a['lat'], row_a['long'], row_b['lat'], row_b['long'])
            if dist > thresh:
                continue
            buffer.append({'a': row_a['text'], 'b': row_b['text'],
                           'i': row_a['orig_index'], 'j': row_b['orig_index']})
            count += 1
            if len(buffer) >= batch_sz:
                process_batch(buffer)
                buffer.clear()
        if count >= max_pairs:
            break

    if buffer:
        process_batch(buffer)

    print(f"Processed {count} pairs, found {sum(trues)} positives")

    # Compute metrics and threshold sweep
    prec, rec, ths = precision_recall_curve(trues, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best = f1s.argmax()
    best_t = ths[best]

    print("\n=== RESULTS ===")
    print(f"Best threshold: {best_t:.3f}")
    print(f"Precision: {prec[best]:.4f}, Recall: {rec[best]:.4f}, F1: {f1s[best]:.4f}")
    acc = accuracy_score(trues, [1 if p>=best_t else 0 for p in probs])
    print(f"Accuracy: {acc:.4f}")

    # Plot confusion matrix
    preds = [1 if p>=best_t else 0 for p in probs]
    cm = confusion_matrix(trues, preds)
    # cm format: [[TN, FP], [FN, TP]]
    # We want columns ordered as [Positive (1), Negative (0)] and rows as [True Positive, True Negative]
    cm_reordered = cm[:, [1, 0]]  # swap columns
    cm_reordered = cm_reordered[[1, 0], :]  # swap rows

    fig, ax = plt.subplots()
    cax = ax.matshow(cm_reordered, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    # Set tick labels in new order
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive (1)', 'Negative (0)'])
    ax.set_yticklabels(['Positive (1)', 'Negative (0)'])
    ax.set_title('Confusion Matrix (Positives first)')
    for (i, j), val in np.ndenumerate(cm_reordered):
        ax.text(j, i, val, ha='center', va='center')
    fig.savefig('confusion_matrix.png')
    
        

    # Save threshold sweep
    out_dir = os.path.join(base_dir, cfg['training']['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        'threshold': ths,
        'precision': prec[:-1],
        'recall': rec[:-1],
        'f1': f1s[:-1]
    }).to_csv(os.path.join(out_dir, 'threshold_sweep.csv'), index=False)

    # Save wrong predictions
    if wrong_records:
        pd.DataFrame(wrong_records).to_csv(os.path.join(out_dir, 'wrong_predictions.csv'), index=False)
        print(f"Wrong predictions saved to: {os.path.join(out_dir, 'wrong_predictions.csv')}")
    else:
        print("No wrong predictions to save.")

