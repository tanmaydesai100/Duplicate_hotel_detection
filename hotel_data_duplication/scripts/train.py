

import os
import re
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertPreTrainedModel, BertModel, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from Levenshtein import ratio
from tqdm.auto import tqdm

class BertJointHead(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        hidden = config.hidden_size
        self.match_head = nn.Linear(hidden + 1, config.num_labels)
        self.idA_head = nn.Linear(hidden, config.num_labels_entity)
        self.idB_head = nn.Linear(hidden, config.num_labels_entity)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, edit_dist=None):
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
        logits_idA = self.idA_head(pooled)
        logits_idB = self.idB_head(pooled)
        return logits_match, logits_idA, logits_idB

class AddressPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a, b = row['text_a'], row['text_b']
        match_label = int(row['label'])
        idA_label = int(row['clusterA'])
        idB_label = int(row['clusterB'])
        sim = ratio(a, b)
        enc = self.tokenizer(
            a, b,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'edit_dist': torch.tensor(sim, dtype=torch.float),
            'match_label': torch.tensor(match_label, dtype=torch.long),
            'idA_label': torch.tensor(idA_label, dtype=torch.long),
            'idB_label': torch.tensor(idB_label, dtype=torch.long)
        }


def collate_fn(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


def train():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg = yaml.safe_load(open(os.path.join(base_dir, 'configs', 'config.yaml')))
    train_csv = os.path.join(base_dir, cfg['data']['input_csv'])
    output_dir = cfg['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(train_csv)
    num_entities = df[['clusterA','clusterB']].max().max() + 1

    tokenizer = BertTokenizerFast.from_pretrained(cfg['model']['name_or_path'])
    bert_conf = BertConfig.from_pretrained(
        cfg['model']['name_or_path'],
        num_labels=int(cfg['model']['num_labels'])
    )
    bert_conf.num_labels_entity = int(num_entities)
    model = BertJointHead(bert_conf).to(device)
    bert_enc = BertModel.from_pretrained(cfg['model']['name_or_path'])
    model.bert.load_state_dict(bert_enc.state_dict())

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_ds = AddressPairDataset(train_df, tokenizer, int(cfg['model']['max_length']))
    val_ds = AddressPairDataset(val_df, tokenizer, int(cfg['model']['max_length']))

    train_dl = DataLoader(
        train_ds,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=False,
        collate_fn=collate_fn
    )

    opt = AdamW(
        model.parameters(),
        lr=float(cfg['training']['learning_rate']),
        weight_decay=float(cfg['training']['weight_decay'])
    )

    epochs = 400
    total_steps = len(train_dl) * epochs
    sched = get_scheduler(
        'linear',
        optimizer=opt,
        num_warmup_steps=int(cfg['training']['warmup_ratio'] * total_steps),
        num_training_steps=total_steps
    )

    lm_start = float(cfg['training'].get('lambda_match_start', 0.5))
    lm_end = float(cfg['training'].get('lambda_match_end', 1.0))
    li_start = float(cfg['training'].get('lambda_id_start', 1.0))
    li_end = float(cfg['training'].get('lambda_id_end', 0.5))

    patience = 30
    best_f1 = 0.0
    patience_counter = 0

    model.train()
    for ep in range(epochs):
        frac = ep / (epochs - 1)
        lm = lm_start + frac * (lm_end - lm_start)
        li = li_start + frac * (li_end - li_start)
        tloss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_match, logits_idA, logits_idB = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                edit_dist=batch['edit_dist']
            )
            loss_m = F.cross_entropy(logits_match, batch['match_label'])
            loss_a = F.cross_entropy(logits_idA, batch['idA_label'])
            loss_b = F.cross_entropy(logits_idB, batch['idB_label'])
            loss = lm * loss_m + li * (loss_a + loss_b)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            tloss += loss.item()

        avg_loss = tloss / len(train_dl)
        print(f"Epoch {ep+1} Avg Loss {avg_loss:.4f} (λm={lm:.3f}, λid={li:.3f})")

        # validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for vb in val_dl:
                vb = {k: v.to(device) for k, v in vb.items()}
                logits_match, _, _ = model(
                    input_ids=vb['input_ids'],
                    attention_mask=vb['attention_mask'],
                    token_type_ids=vb['token_type_ids'],
                    edit_dist=vb['edit_dist']
                )
                preds = torch.argmax(logits_match, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(vb['match_label'].cpu().tolist())
        val_f1 = f1_score(val_labels, val_preds)
        print(f"Validation F1 after epoch {ep+1}: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("✓ Saved best model")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                return
        model.train()

if __name__ == '__main__':
    train()

