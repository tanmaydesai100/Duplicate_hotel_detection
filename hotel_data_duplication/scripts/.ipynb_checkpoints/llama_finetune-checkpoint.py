
# import os
# import gc
# # Aggressive memory optimization
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8,expandable_segments:True"

# import yaml
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from torch.optim import AdamW
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     AutoTokenizer,
#     LlamaConfig,
#     LlamaForSequenceClassification,
#     get_scheduler,
# )
# from transformers.modeling_outputs import SequenceClassifierOutput
# from peft import LoraConfig, get_peft_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from tqdm.auto import tqdm
# from Levenshtein import ratio

# # Try to import bitsandbytes, fallback if not available
# try:
#     from transformers import BitsAndBytesConfig
#     from peft import prepare_model_for_kbit_training
#     QUANTIZATION_AVAILABLE = True
#     print("✓ BitsAndBytes available - will use 4-bit quantization")
# except ImportError:
#     QUANTIZATION_AVAILABLE = False
#     print("⚠ BitsAndBytes not available - using standard precision")


# def print_memory_stats(stage=""):
#     """Print GPU memory statistics"""
#     if torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated() / 1024**3
#         reserved = torch.cuda.memory_reserved() / 1024**3
#         max_allocated = torch.cuda.max_memory_allocated() / 1024**3
#         print(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")


# def clear_memory():
#     """Aggressive memory clearing"""
#     gc.collect()
#     torch.cuda.empty_cache()
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()


# def load_config():
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg_path = os.path.join(base_dir, 'configs', 'config.yaml')
#     return yaml.safe_load(open(cfg_path, 'r'))


# class SentencePairDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len):
#         self.df = df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self): return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         a, b = row['text_a'], row['text_b']
#         label = int(row['label'])
#         edit_dist = ratio(a, b)
#         enc = self.tokenizer(
#             a, b,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_len,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': enc['input_ids'].squeeze(0),
#             'attention_mask': enc['attention_mask'].squeeze(0),
#             'labels': torch.tensor(label, dtype=torch.long),
#             'edit_dist': torch.tensor(edit_dist, dtype=torch.float)
#         }


# def collate_fn(batch): return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


# class LlamaJointHead(LlamaForSequenceClassification):
#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)
#         hidden = config.hidden_size
#         self.score = torch.nn.Linear(hidden + 1, config.num_labels)

#     def forward(self, input_ids=None, attention_mask=None, labels=None, edit_dist=None, **kwargs):
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             **kwargs
#         )
#         last_hidden = outputs.last_hidden_state
#         eos_mask = (input_ids == self.config.eos_token_id).unsqueeze(-1)
#         pooled = (last_hidden * eos_mask).sum(1)
#         ed = edit_dist.unsqueeze(1).to(pooled.dtype)
#         feats = torch.cat([pooled, ed], dim=1)
#         logits = self.score(feats)
#         loss = None
#         if labels is not None:
#             loss = torch.nn.CrossEntropyLoss()(logits, labels)
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions
#         )


# def load_model_with_quantization(model_name, config, max_memory_gb=35):
#     """Load model with or without quantization based on availability"""
    
#     if QUANTIZATION_AVAILABLE:
#         print("Loading model with 4-bit quantization...")
#         quant_cfg = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4"
#         )
        
#         model = LlamaJointHead.from_pretrained(
#             model_name,
#             config=config,
#             quantization_config=quant_cfg,
#             device_map='auto',
#             low_cpu_mem_usage=True,
#             max_memory={0: f"{max_memory_gb}GB"}
#         )
        
#         # Prepare model for k-bit training
#         model = prepare_model_for_kbit_training(model)
        
#     else:
#         print("Loading model with standard precision...")
#         model = LlamaJointHead.from_pretrained(
#             model_name,
#             config=config,
#             torch_dtype=torch.float16,  # Use float16 to save memory
#             device_map='auto',
#             low_cpu_mem_usage=True,
#             max_memory={0: f"{max_memory_gb}GB"}
#         )
    
#     return model


# if __name__ == '__main__':
#     print_memory_stats("Initial")
    
#     cfg = load_config()
    
#     # Adjust batch size and sequence length based on available memory
#     if QUANTIZATION_AVAILABLE:
#         batch_size = min(cfg['training'].get('batch_size', 8), 4)  # Can use slightly larger batch
#         max_length = min(cfg['model'].get('max_length', 512), 512)
#     else:
#         batch_size = min(cfg['training'].get('batch_size', 8), 2)  # Smaller batch for full precision
#         max_length = min(cfg['model'].get('max_length', 512), 256)  # Shorter sequences
    
#     print(f"Using batch_size: {batch_size}, max_length: {max_length}")
    
#     df = pd.read_csv(cfg['data']['input_csv'])
#     train_df, val_df = train_test_split(
#         df,
#         test_size=float(cfg['data'].get('test_size', 0.1)),
#         random_state=int(cfg['data'].get('random_seed', 42))
#     )

#     print("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name_or_path'], use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     print_memory_stats("After tokenizer")
#     clear_memory()

#     # prepare model config
#     llm_conf = LlamaConfig.from_pretrained(
#         cfg['model']['name_or_path'],
#         num_labels=int(cfg['model']['num_labels'])
#     )

#     # Load model with appropriate precision
#     model = load_model_with_quantization(
#         cfg['model']['name_or_path'], 
#         llm_conf, 
#         max_memory_gb=35
#     )
    
#     print_memory_stats("After model loading")
    
#     # Setup LoRA
#     lora_rank = 8 if QUANTIZATION_AVAILABLE else 4  # Smaller rank for full precision
#     lora_cfg = LoraConfig(
#         r=min(int(cfg['lora']['rank']), lora_rank),
#         lora_alpha=int(cfg['lora']['alpha']),
#         target_modules=cfg['lora']['target_modules'],
#         lora_dropout=float(cfg['lora']['dropout']),
#         bias='none',
#         task_type='SEQ_CLS'
#     )
#     model = get_peft_model(model, lora_cfg)
    
#     print_memory_stats("After LoRA")

#     model.config.pad_token_id = tokenizer.pad_token_id
    
#     # Ensure model parameters require gradients
#     for name, param in model.named_parameters():
#         if 'lora' in name.lower() or 'score' in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
    
#     # Print parameter info
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Trainable parameters: {trainable_params:,}")
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
#     clear_memory()
#     print_memory_stats("After setup")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     train_dl = DataLoader(
#         SentencePairDataset(train_df, tokenizer, max_length),
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_fn,
#         pin_memory=False,
#         num_workers=0
#     )
#     val_dl = DataLoader(
#         SentencePairDataset(val_df, tokenizer, max_length),
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_fn,
#         pin_memory=False,
#         num_workers=0
#     )

#     # Only trainable parameters
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
    
#     optimizer = AdamW(
#         trainable_params,
#         lr=float(cfg['training']['learning_rate']),
#         weight_decay=float(cfg['training']['weight_decay'])
#     )
    
#     total_steps = len(train_dl) * int(cfg['training']['epochs'])
#     scheduler = get_scheduler(
#         'linear',
#         optimizer=optimizer,
#         num_warmup_steps=int(float(cfg['training']['warmup_ratio']) * total_steps),
#         num_training_steps=total_steps
#     )

#     # Adjust gradient accumulation based on precision
#     base_accumulation = cfg['training'].get('gradient_accumulation_steps', 4)
#     accumulation_steps = base_accumulation if QUANTIZATION_AVAILABLE else max(base_accumulation, 8)
#     print(f"Using gradient accumulation steps: {accumulation_steps}")

#     best_f1, patience_cnt = 0.0, 0
#     for epoch in range(1, int(cfg['training']['epochs']) + 1):
#         model.train()
#         total_loss = 0.0
        
#         clear_memory()
#         print_memory_stats(f"Epoch {epoch} start")
        
#         for batch_idx, batch in enumerate(tqdm(train_dl, desc=f"Train {epoch}/{cfg['training']['epochs']}")):
#             batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
#             # Forward pass
#             out = model(
#                 input_ids=batch['input_ids'],
#                 attention_mask=batch['attention_mask'],
#                 labels=batch['labels'],
#                 edit_dist=batch['edit_dist']
#             )
            
#             loss = out.loss / accumulation_steps
#             loss.backward()
            
#             if (batch_idx + 1) % accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 scheduler.step()
            
#             total_loss += out.loss.item()
            
#             # Clear memory more frequently for full precision
#             clear_freq = 25 if QUANTIZATION_AVAILABLE else 10
#             if batch_idx % clear_freq == 0:
#                 clear_memory()
                
#         print(f"Epoch {epoch} loss: {total_loss/len(train_dl):.4f}")
#         print_memory_stats(f"Epoch {epoch} end")

#         # Validation
#         model.eval()
#         preds, truth = [], []
#         with torch.no_grad():
#             for batch in tqdm(val_dl, desc="Validating"):
#                 batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
#                 out = model(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask'],
#                     edit_dist=batch['edit_dist']
#                 )
#                 logits = out.logits
#                 preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
#                 truth.extend(batch['labels'].cpu().tolist())
                
#         f1 = f1_score(truth, preds)
#         print(f"Epoch {epoch} Validation F1: {f1:.4f}")

#         if f1 > best_f1:
#             best_f1, patience_cnt = f1, 0
#             os.makedirs(cfg['training']['output_dir'], exist_ok=True)
#             model.save_pretrained(cfg['training']['output_dir'])
#             tokenizer.save_pretrained(cfg['training']['output_dir'])
#             print("Saved best model")
#         else:
#             patience_cnt += 1
#             if patience_cnt >= int(cfg['training']['patience']):
#                 print("Early stopping")
#                 break
                
#         clear_memory()
        
#     print(f"Training completed. Best F1: {best_f1:.4f}")
#     print("Final memory cleanup...")
#     clear_memory()
#     print_memory_stats("Final")


# not working llama

# import os
# import gc
# import yaml
# import pandas as pd
# import torch
# from torch.optim import AdamW
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     AutoTokenizer,
#     LlamaConfig,
#     LlamaForCausalLM,
#     get_scheduler,
# )
# from peft import LoraConfig, get_peft_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from tqdm.auto import tqdm
# from Levenshtein import ratio

# # Mixed-precision via torch.amp
# from torch.amp import autocast, GradScaler

# # Check for BitsAndBytes for quantization (optional)
# try:
#     from transformers import BitsAndBytesConfig
#     from peft import prepare_model_for_kbit_training
#     QUANTIZATION_AVAILABLE = True
#     print("\u2713 BitsAndBytes available - will use 8-bit quantization")
# except ImportError:
#     QUANTIZATION_AVAILABLE = False
#     print("\u26a0 BitsAndBytes not available - using standard precision")


# def load_config():
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg_path = os.path.join(base_dir, 'configs', 'config.yaml')
#     return yaml.safe_load(open(cfg_path, 'r'))


# class SentencePairDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len):
#         self.df = df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         a, b = row['text_a'], row['text_b']
#         label = int(row['label'])
#         edit_dist = ratio(a, b)

#         prompt = f"Sentence 1: {a}\nSentence 2: {b}\nAnswer the question: Are these sentences similar or different?"

#         enc = self.tokenizer(
#             prompt,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_len,
#             return_tensors='pt'
#         )

#         return {
#             'input_ids': enc['input_ids'].squeeze(0),
#             'attention_mask': enc['attention_mask'].squeeze(0),
#             'labels': torch.tensor(label, dtype=torch.long),
#             'edit_dist': torch.tensor(edit_dist, dtype=torch.float)
#         }


# def collate_fn(batch):
#     return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


# class LlamaWithGeneration(LlamaForCausalLM):
#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)
#         num_labels = getattr(config, "num_labels", 2)
#         self.score = torch.nn.Linear(config.hidden_size, num_labels)

#     def forward(self,
#                 input_ids=None,
#                 attention_mask=None,
#                 labels=None,
#                 **kwargs):
#         outputs = super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=None,
#             output_hidden_states=True,
#             return_dict=True
#         )

#         last_hidden = outputs.hidden_states[-1]  # [B, T, H]
#         mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())  # [B, T, H]
#         masked_hidden = last_hidden * mask
#         pooled = masked_hidden.sum(1) / mask.sum(1).clamp(min=1e-8)  # [B, H]

#         logits = self.score(pooled)  # [B, num_labels]

#         loss = None
#         if labels is not None:
#             loss_fct = torch.nn.CrossEntropyLoss()
#             loss = loss_fct(logits, labels)

#         return {"loss": loss, "logits": logits}

#     def generate_text(self, input_ids, attention_mask, **gen_kwargs):
#         return super().generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             **gen_kwargs
#         )


# def load_model_with_quantization(model_name, config, max_memory_gb=35):
#     if QUANTIZATION_AVAILABLE:
#         quant_cfg = BitsAndBytesConfig(
#             load_in_8bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4"
#         )
#         model = LlamaWithGeneration.from_pretrained(
#             model_name,
#             config=config,
#             quantization_config=quant_cfg,
#             device_map='auto',
#             low_cpu_mem_usage=True,
#             max_memory={0: f"{max_memory_gb}GB"}
#         )
#         model = prepare_model_for_kbit_training(model)
#     else:
#         model = LlamaWithGeneration.from_pretrained(
#             model_name,
#             config=config,
#             torch_dtype=torch.float16,
#             device_map='auto',
#             low_cpu_mem_usage=True,
#             max_memory={0: f"{max_memory_gb}GB"}
#         )
#     return model


# def load_model_without_quantization(model_name, config, max_memory_gb=35):
#     print("Loading model with standard precision (no quantization)...")
#     return LlamaWithGeneration.from_pretrained(
#         model_name,
#         config=config,
#         torch_dtype=torch.float16,
#         device_map='auto',
#         low_cpu_mem_usage=True,
#         max_memory={0: f"{max_memory_gb}GB"}
#     )


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cfg = load_config()

#     batch_size = min(cfg['training'].get('batch_size', 8),
#                      4 if QUANTIZATION_AVAILABLE else 2)
#     max_length = min(cfg['model'].get('max_length', 512),
#                      256 if QUANTIZATION_AVAILABLE else 512)

#     df = pd.read_csv(cfg['data']['input_csv'])
#     df = df.dropna(subset=['text_a', 'text_b', 'label'])
#     df = df[df['label'].isin([0, 1])]

#     train_df, val_df = train_test_split(
#         df,
#         test_size=float(cfg['data'].get('test_size', 0.1)),
#         random_state=int(cfg['data'].get('random_seed', 42))
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#         cfg['model']['name_or_path'], use_fast=True
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     llm_conf = LlamaConfig.from_pretrained(
#         cfg['model']['name_or_path'],
#         num_labels=int(cfg['model']['num_labels'])
#     )

#     model = load_model_without_quantization(
#         cfg['model']['name_or_path'], llm_conf, max_memory_gb=35
#     )
#     model.gradient_checkpointing_enable()
#     model.to(device)
#     # Ensure classification head matches model dtype
#     model.score = model.score.to(device=device, dtype=next(model.parameters()).dtype)

#     lora_cfg = LoraConfig(
#         r=min(int(cfg['lora']['rank']),
#               8 if QUANTIZATION_AVAILABLE else 4),
#         lora_alpha=int(cfg['lora']['alpha']),
#         target_modules=cfg['lora']['target_modules'],
#         lora_dropout=float(cfg['lora']['dropout']),
#         bias='none',
#         task_type='SEQ_CLS'
#     )
#     model = get_peft_model(model, lora_cfg)
#     # Ensure entire model (including classification head) is in FP16 if using mixed precision
#     if cfg['training'].get('fp16', False):
#         model = model.half()
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.pad_token_id = tokenizer.pad_token_id

#     # Optimizer and AMP setup
#     optimizer = AdamW(
#         [p for p in model.parameters() if p.requires_grad],
#         lr=float(cfg['training']['learning_rate']),
#         weight_decay=float(cfg['training']['weight_decay'])
#     )
#     scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

#     # DataLoaders
#     train_dl = DataLoader(
#         SentencePairDataset(train_df, tokenizer, max_length),
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_fn
#     )
#     val_dl = DataLoader(
#         SentencePairDataset(val_df, tokenizer, max_length),
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_fn
#     )

#     # Scheduler setup
#     total_steps = len(train_dl) * int(cfg['training']['epochs'])
#     scheduler = get_scheduler(
#         name=cfg['training'].get('scheduler', 'linear'),
#         optimizer=optimizer,
#         num_warmup_steps=int(cfg['training'].get('warmup_steps', 0)),
#         num_training_steps=total_steps
#     )

#     best_f1, patience_cnt = 0.0, 0
#     accumulation_steps = int(cfg['training'].get('gradient_accumulation_steps', 1))

#     for epoch in range(1, int(cfg['training']['epochs']) + 1):
#         model.train()
#         total_loss = 0.0

#         for batch_idx, batch in enumerate(tqdm(train_dl, desc=f"Train {epoch}")):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
#                 out = model(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask'],
#                     labels=batch['labels']
#                 )
#                 loss = out['loss'] / accumulation_steps

#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             if (batch_idx + 1) % accumulation_steps == 0:
#                 try:
#                     scaler.step(optimizer)
#                 except ValueError:
#                     optimizer.step()
#                 try:
#                     scaler.update()
#                 except AssertionError:
#                     pass
#                 optimizer.zero_grad()
#                 scheduler.step()

#             total_loss += loss.item()

#         model.eval()
#         preds, truth = [], []
#         with torch.no_grad():
#             for batch in tqdm(val_dl, desc="Validating"):
#                 batch = {k: v.to(device) for k, v in batch.items()}
#                 out = model(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask']
#                 )
#                 preds.extend(torch.argmax(out['logits'], dim=-1).cpu().tolist())
#                 truth.extend(batch['labels'].cpu().tolist())

#         f1 = f1_score(truth, preds)
#         print(f"Epoch {epoch} — Train loss: {total_loss/len(train_dl):.4f}, Val F1: {f1:.4f}")

#         if f1 > best_f1:
#             best_f1, patience_cnt = f1, 0
#             model.save_pretrained(cfg['training']['output_dir'])
#             tokenizer.save_pretrained(cfg['training']['output_dir'])
#         else:
#             patience_cnt += 1
#             if patience_cnt >= int(cfg['training']['patience']):
#                 print("Early stopping.")
#                 break

#     print(f"Training completed. Best F1: {best_f1:.4f}")










import os
import gc
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
import json

# Try to import bitsandbytes for quantization
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
    print("✓ BitsAndBytes available - will use 4-bit quantization")
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("⚠ BitsAndBytes not available - using standard precision")

# Aggressive memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8,expandable_segments:True"

def print_memory_stats(stage=""):
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_config():
    """Load config file"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_path = os.path.join(base_dir, 'configs', 'config.yaml')
    return yaml.safe_load(open(cfg_path, 'r'))

class AddressMatchingDataset(Dataset):
    """Dataset for generative address matching fine-tuning"""
    
    def __init__(self, df, tokenizer, max_length=512, prompt_template="instruct"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        
class AddressMatchingDataset(Dataset):
    """Dataset for generative address matching fine-tuning"""
    
    def __init__(self, df, tokenizer, max_length=512, prompt_template="instruct"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        
    def create_instruction_prompt(self, address_a, address_b, label=None, include_answer=True):
        """Create instruction-following prompt for hotel/address matching"""
        
        if self.prompt_template == "instruct":
            system_prompt = """You are an expert at hotel and address matching. Your task is to determine if two hotel listings or addresses refer to the same location. Consider variations in formatting, abbreviations, hotel names, and minor differences that don't change the actual location.

Instructions:
- Compare the two entries carefully
- Consider hotel name variations and abbreviations
- Consider address formatting differences (St/Street, Ave/Avenue, etc.)
- Consider location and geographical proximity
- Ignore case differences and extra spaces
- Answer with exactly "MATCH" if they refer to the same hotel/location, or "NO_MATCH" if they don't"""

            user_prompt = f"""Entry 1: {address_a}
Entry 2: {address_b}

Do these entries match?"""

            if include_answer and label is not None:
                answer = "MATCH" if label == 1 else "NO_MATCH"
                conversation = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""
            else:
                conversation = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        elif self.prompt_template == "few_shot":
            # Few-shot examples for hotel matching
            examples = """Examples:
Entry 1: Marriott Hotel, 123 Main St, New York, NY 10001
Entry 2: Marriott NYC, 123 Main Street, New York, NY 10001
Answer: MATCH

Entry 1: Holiday Inn Express, 456 Oak Avenue, Los Angeles, CA 90210
Entry 2: Hilton Garden Inn, 789 Pine Road, Chicago, IL 60601
Answer: NO_MATCH

Entry 1: Best Western Downtown, 100 First Ave
Entry 2: Best Western Hotel, 100 1st Avenue
Answer: MATCH"""

            conversation = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a hotel and address matching expert. Determine if two entries refer to the same hotel/location.

{examples}<|eot_id|><|start_header_id|>user<|end_header_id|>

Entry 1: {address_a}
Entry 2: {address_b}
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            
            if include_answer and label is not None:
                answer = "MATCH" if label == 1 else "NO_MATCH"
                conversation += f"{answer}<|eot_id|>"
        
        return conversation
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        address_a = str(row['text_a']).strip()
        address_b = str(row['text_b']).strip()
        label = int(row['label'])
        
        # Create full conversation with answer for training
        full_prompt = self.create_instruction_prompt(address_a, address_b, label, include_answer=True)
        
        # Tokenize
        encoding = self.tokenizer(
            full_prompt,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # For causal LM, labels are the same as input_ids (shifted internally by the model)
        labels = input_ids.clone()
        
        # Mask the system and user parts, only train on assistant response
        # Find assistant response start
        assistant_start_token = "<|start_header_id|>assistant<|end_header_id|>"
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        if assistant_start_token in full_text:
            # Find where assistant response starts
            assistant_start_pos = full_text.find(assistant_start_token) + len(assistant_start_token)
            assistant_tokens = self.tokenizer(full_text[assistant_start_pos:], add_special_tokens=False)['input_ids']
            
            # Mask everything except assistant response
            mask_length = len(input_ids) - len(assistant_tokens)
            labels[:mask_length] = -100  # Ignore in loss calculation
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def collate_fn(batch):
    """Custom collate function for padding"""
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Pad sequences
        pad_length = max_len - len(item['input_ids'])
        
        padded_input_ids = F.pad(item['input_ids'], (0, pad_length), value=0)  # PAD token
        padded_attention_mask = F.pad(item['attention_mask'], (0, pad_length), value=0)
        padded_labels = F.pad(item['labels'], (0, pad_length), value=-100)  # Ignore padded tokens
        
        input_ids.append(padded_input_ids)
        attention_masks.append(padded_attention_mask)
        labels.append(padded_labels)
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }

class AddressMatchingEvaluator:
    """Evaluator for address matching task"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def predict_single(self, address_a, address_b, max_new_tokens=10):
        """Predict single address pair"""
        # Create prompt without answer
        dataset = AddressMatchingDataset(
            pd.DataFrame([{'text_a': address_a, 'text_b': address_b, 'label': 0}]),
            self.tokenizer
        )
        prompt = dataset.create_instruction_prompt(address_a, address_b, include_answer=False)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated part
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip().upper()
        
        # Parse prediction
        if "MATCH" in generated and "NO_MATCH" not in generated:
            return 1, generated
        elif "NO_MATCH" in generated:
            return 0, generated
        else:
            # Fallback heuristic
            return 0, generated
    
    def evaluate_dataset(self, df, sample_size=None):
        """Evaluate on dataset"""
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        predictions = []
        true_labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            pred, generated = self.predict_single(row['text_a'], row['text_b'])
            predictions.append(pred)
            true_labels.append(row['label'])
            
            if idx < 3:  # Show first few examples
                print(f"\nExample {idx + 1}:")
                print(f"  Address A: {row['text_a']}")
                print(f"  Address B: {row['text_b']}")
                print(f"  True: {row['label']}, Predicted: {pred}")
                print(f"  Generated: '{generated}'")
        
        return predictions, true_labels

def load_model_with_quantization(model_name, cfg, max_memory_gb=40):
    """Load model with quantization based on config"""
    
    use_4bit = cfg['training'].get('use_4bit', False)
    use_8bit = cfg['training'].get('use_8bit', False)
    use_fp16 = cfg['training'].get('fp16', False)
    
    if QUANTIZATION_AVAILABLE and use_4bit:
        print("Loading model with 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if not use_fp16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map='auto',
            torch_dtype=torch.bfloat16 if not use_fp16 else torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: f"{max_memory_gb}GB"}
        )
        
        model = prepare_model_for_kbit_training(model)
        
    elif QUANTIZATION_AVAILABLE and use_8bit:
        print("Loading model with 8-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            int8_threshold=cfg['training'].get('int8_threshold', 6.0)
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map='auto',
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: f"{max_memory_gb}GB"}
        )
        
        model = prepare_model_for_kbit_training(model)
        
    else:
        print("Loading model with standard precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: f"{max_memory_gb}GB"}
        )
    
    return model

if __name__ == '__main__':
    print_memory_stats("Initial")
    
    # Load config and data
    cfg = load_config()
    
    # Read CSV from config
    df = pd.read_csv(cfg['data']['input_csv'])
    print(f"Loaded {len(df)} address pairs from {cfg['data']['input_csv']}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Split data
    train_df, val_df = train_test_split(
        df,
        test_size=float(cfg['data'].get('test_size', 0.1)),
        random_state=int(cfg['data'].get('random_seed', 42)),
        stratify=df['label']  # Maintain label distribution
    )
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")
    
    # Load tokenizer
    model_name = cfg['model']['name_or_path']
    max_length = int(cfg['model']['max_length'])
    print(f"Loading tokenizer for {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print_memory_stats("After tokenizer")
    clear_memory()
    
    # Load model
    model = load_model_with_quantization(model_name, cfg)
    print_memory_stats("After model loading")
    
    # Setup LoRA with config parameters
    lora_config = LoraConfig(
        r=int(cfg['lora']['rank']),
        lora_alpha=int(cfg['lora']['alpha']),
        target_modules=cfg['lora']['target_modules'],
        lora_dropout=float(cfg['lora']['dropout']),
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print_memory_stats("After LoRA")
    
    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Create datasets
    train_dataset = AddressMatchingDataset(train_df, tokenizer, max_length=max_length, prompt_template="instruct")
    val_dataset = AddressMatchingDataset(val_df, tokenizer, max_length=max_length, prompt_template="instruct")
    
    # Training parameters from config
    batch_size = int(cfg['training']['batch_size'])
    gradient_accumulation_steps = int(cfg['training']['gradient_accumulation_steps'])
    
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # Setup training with config parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg['training']['learning_rate']),
        weight_decay=float(cfg['training']['weight_decay'])
    )
    
    epochs = int(cfg['training']['epochs'])
    warmup_ratio = float(cfg['training']['warmup_ratio'])
    
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    best_f1 = 0.0
    patience_counter = 0
    patience = int(cfg['training']['patience'])
    logging_steps = int(cfg['training']['logging_steps'])
    eval_steps = int(cfg['training']['eval_steps'])
    
    evaluator = AddressMatchingEvaluator(model, tokenizer, device)
    step_count = 0
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        optimizer.zero_grad()
        
        clear_memory()
        print_memory_stats(f"Epoch {epoch} start")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                
                # Logging
                if step_count % logging_steps == 0:
                    print(f"Step {step_count}, Loss: {outputs.loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                
                # Evaluation
                if step_count % eval_steps == 0:
                    print(f"Running evaluation at step {step_count}...")
                    model.eval()
                    
                    # Evaluate on a sample to save time during training
                    sample_val_df = val_df.sample(n=min(20, len(val_df)), random_state=42)
                    predictions, true_labels = evaluator.evaluate_dataset(sample_val_df)
                    
                    f1 = f1_score(true_labels, predictions, average='weighted')
                    acc = accuracy_score(true_labels, predictions)
                    
                    print(f"Step {step_count} - Validation F1: {f1:.4f}, Accuracy: {acc:.4f}")
                    
                    # Save best model
                    if f1 > best_f1:
                        best_f1 = f1
                        patience_counter = 0
                        
                        output_dir = cfg['training']['output_dir']
                        os.makedirs(output_dir, exist_ok=True)
                        
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        
                        # Save config for reproducibility
                        with open(os.path.join(output_dir, 'training_config.yaml'), 'w') as f:
                            yaml.dump(cfg, f)
                        
                        print(f"Saved best model with F1: {best_f1:.4f}")
                    else:
                        patience_counter += 1
                    
                    model.train()
                
                # Clear memory periodically
                if step_count % (eval_steps // 2) == 0:
                    clear_memory()
            
            total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # Check early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
        
        clear_memory()
    
    print(f"Training completed! Best F1: {best_f1:.4f}")
    
    # Final evaluation on full validation set
    print("\nRunning final evaluation on full validation set...")
    model.eval()
    final_predictions, final_true_labels = evaluator.evaluate_dataset(val_df)
    final_f1 = f1_score(final_true_labels, final_predictions, average='weighted')
    final_acc = accuracy_score(final_true_labels, final_predictions)
    
    print(f"Final Results - F1: {final_f1:.4f}, Accuracy: {final_acc:.4f}")
    
    # Save final results
    results = {
        'best_validation_f1': float(best_f1),
        'final_validation_f1': float(final_f1),
        'final_validation_accuracy': float(final_acc),
        'total_training_steps': step_count,
        'epochs_completed': epoch
    }
    
    with open(os.path.join(cfg['training']['output_dir'], 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Results saved to: {cfg['training']['output_dir']}")
    
    clear_memory()
    print_memory_stats("Final")