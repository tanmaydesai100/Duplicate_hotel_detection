


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

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✓ wandb available - will log training to Weights & Biases")
except Exception:
    wandb = None
    WANDB_AVAILABLE = False
    print("⚠ wandb not available - continuing without W&B logging")

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
    
    def evaluate_dataset(self, df, sample_size=None, log_to_wandb=False):
        """Evaluate on dataset"""
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        predictions = []
        true_labels = []
        examples_for_log = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            pred, generated = self.predict_single(row['text_a'], row['text_b'])
            predictions.append(pred)
            true_labels.append(row['label'])
            
            if idx < 10:  # collect some examples for logging
                examples_for_log.append({
                    'address_a': row['text_a'],
                    'address_b': row['text_b'],
                    'true': int(row['label']),
                    'pred': int(pred),
                    'generated': generated
                })
                
            if idx < 3:  # Show first few examples on stdout
                print(f"\nExample {idx + 1}:")
                print(f"  Address A: {row['text_a']}")
                print(f"  Address B: {row['text_b']}")
                print(f"  True: {row['label']}, Predicted: {pred}")
                print(f"  Generated: '{generated}'")
        
        if log_to_wandb and WANDB_AVAILABLE:
            # Create a W&B Table with examples
            try:
                table = wandb.Table(columns=["address_a", "address_b", "true", "pred", "generated"])
                for e in examples_for_log:
                    table.add_data(e['address_a'], e['address_b'], e['true'], e['pred'], e['generated'])
                wandb.log({"eval/examples": table})
            except Exception as e:
                print("Warning: failed to log examples to W&B:", e)
        
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
    
    # Initialize W&B if available and enabled in config
    wandb_enabled = WANDB_AVAILABLE and cfg['training'].get('use_wandb', True)
    if wandb_enabled:
        try:
            wandb_project = cfg['training'].get('wandb_project', 'address-matching')
            wandb_run_name = cfg['training'].get('run_name', None)
            wandb.init(project=wandb_project, config=cfg, name=wandb_run_name, reinit=True)
            # Watch the model (will try to log gradients & parameters)
            # Note: for very large models or k-bit/PEFT setups this can be noisy; adjust as needed.
            # wandb.watch will be called after model is moved to device / PEFT applied.
            print(f"W&B initialized: project={wandb_project}, run_name={wandb_run_name}")
        except Exception as e:
            print("Failed to initialize W&B, continuing without it:", e)
            wandb_enabled = False
    
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

    # If W&B enabled, watch the model now (after PEFT wrapping)
    if wandb_enabled:
        try:
            wandb.watch(model, log='all', log_freq=100)
        except Exception as e:
            print("Warning: wandb.watch failed:", e)
    
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
                    lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else None
                    print(f"Step {step_count}, Loss: {outputs.loss.item():.4f}, LR: {lr:.2e}")
                    if wandb_enabled:
                        try:
                            wandb.log({
                                'train/loss': outputs.loss.item(),
                                'train/lr': lr,
                                'train/step': step_count,
                                'epoch': epoch
                            }, step=step_count)
                        except Exception as e:
                            print("Warning: failed to log training metrics to W&B:", e)
                
                # Evaluation
                if step_count % eval_steps == 0:
                    print(f"Running evaluation at step {step_count}...")
                    model.eval()
                    
                    # Evaluate on a sample to save time during training
                    sample_val_df = val_df.sample(n=min(20, len(val_df)), random_state=42)
                    predictions, true_labels = evaluator.evaluate_dataset(sample_val_df, log_to_wandb=wandb_enabled)
                    
                    f1 = f1_score(true_labels, predictions, average='weighted')
                    acc = accuracy_score(true_labels, predictions)
                    
                    print(f"Step {step_count} - Validation F1: {f1:.4f}, Accuracy: {acc:.4f}")
                    
                    # Log evaluation metrics to W&B
                    if wandb_enabled:
                        try:
                            wandb.log({
                                'val/f1': f1,
                                'val/acc': acc,
                                'val/step': step_count,
                                'epoch': epoch
                            }, step=step_count)
                        except Exception as e:
                            print("Warning: failed to log validation metrics to W&B:", e)
                    
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
                        
                        # Log model artifact to W&B
                        if wandb_enabled:
                            try:
                                artifact = wandb.Artifact(name=f"best-model-{wandb.run.id}", type="model")
                                artifact.add_dir(output_dir)
                                wandb.log_artifact(artifact)
                                print("Uploaded model artifact to W&B")
                            except Exception as e:
                                print("Warning: failed to upload model artifact to W&B:", e)
                    else:
                        patience_counter += 1
                    
                    model.train()
                
                # Clear memory periodically
                if eval_steps > 0 and (step_count % max(1, (eval_steps // 2)) == 0):
                    clear_memory()
            
            total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        if wandb_enabled:
            try:
                wandb.log({'train/epoch_loss': avg_loss, 'epoch': epoch}, step=step_count)
            except Exception as e:
                print("Warning: failed to log epoch metrics to W&B:", e)
        
        # Check early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
        
        clear_memory()
    
    print(f"Training completed! Best F1: {best_f1:.4f}")
    
    # Final evaluation on full validation set
    print("\nRunning final evaluation on full validation set...")
    model.eval()
    final_predictions, final_true_labels = evaluator.evaluate_dataset(val_df, log_to_wandb=wandb_enabled)
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
    
    # If W&B enabled, log final metrics and finish run
    if wandb_enabled:
        try:
            wandb.log({
                'final/f1': final_f1,
                'final/acc': final_acc,
                'best/f1': best_f1,
                'total_steps': step_count,
                'epochs_completed': epoch
            }, step=step_count)
            wandb.finish()
            print("W&B run finished")
        except Exception as e:
            print("Warning: failed to finalize W&B run:", e)
    
    clear_memory()
    print_memory_stats("Final")



