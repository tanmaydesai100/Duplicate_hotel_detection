

# all the pairs

import os
import gc
import yaml
import pandas as pd
import numpy as np
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm.auto import tqdm
import json
from datetime import datetime
import random

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def print_memory_stats(stage=""):
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")

def load_config():
    """Load config file"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_path = os.path.join(base_dir, 'configs', 'config.yaml')
    return yaml.safe_load(open(cfg_path, 'r'))

# Haversine distance function (meters) - same as BERT code
def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

class AddressEvaluator:
    """LLaMA evaluator with exhaustive pairwise comparison like BERT"""
    
    def __init__(self, model_path, base_model_name, device='cuda'):
        self.device = device
        self.model_path = model_path
        self.base_model_name = base_model_name
        
        print(f"Loading tokenizer from {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Loading base model from {base_model_name}...")
        # Load base model with same configuration as training
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quant_config,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            print("✓ Loaded base model with 4-bit quantization")
        except ImportError:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
            )
            print("✓ Loaded base model with float16")
        
        print(f"Loading LoRA adapter from {model_path}...")
        self.model = PeftModel.from_pretrained(self.model, model_path)
        print("✓ LoRA adapter loaded successfully")
        
        self.model.eval()
        print_memory_stats("After model loading")
    
    def create_evaluation_prompt(self, address_a, address_b):
        """Create the same prompt format used during training"""
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

        conversation = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return conversation
    
    def predict_single_pair(self, address_a, address_b, max_new_tokens=10):
        """Predict whether two addresses match"""
        try:
            # Create prompt
            prompt = self.create_evaluation_prompt(address_a, address_b)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip().upper()
            
            # Parse prediction
            if "MATCH" in generated_text and "NO_MATCH" not in generated_text:
                return 1, generated_text
            elif "NO_MATCH" in generated_text:
                return 0, generated_text
            else:
                # If unclear, return 0 (no match) as default
                return 0, generated_text
                
        except Exception as e:
            print(f"Error processing pair: {e}")
            return 0, "ERROR"
    
    def evaluate_exhaustive_pairs(self, addresses_df, inference_pairs_df=None, 
                                 distance_threshold_m=600, max_pairs=float('inf'), 
                                 save_predictions=True, batch_size=50):
        """
        Exhaustive pairwise evaluation like BERT code:
        - Compare every address to every other address
        - Apply Haversine distance filtering
        - Use inference_pairs_df as ground truth for labeling
        """
        
        print(f"Starting exhaustive pairwise evaluation...")
        print(f"Total addresses: {len(addresses_df)}")
        print(f"Distance threshold: {distance_threshold_m}m")
        print(f"Max pairs to evaluate: {max_pairs}")
        
        # Create ground truth set from inference pairs (like BERT code)
        dup_set = set()
        if inference_pairs_df is not None:
            for _, row in inference_pairs_df.iterrows():
                pair = frozenset((row['orig_indexA'], row['orig_indexB']))
                dup_set.add(pair)
            print(f"Ground truth duplicates: {len(dup_set)}")
        
        # Prepare results storage
        results = []
        predictions = []
        true_labels = []
        wrong_records = []
        
        count = 0
        processed_pairs = 0
        skipped_by_distance = 0
        
        print("Generating and evaluating candidate pairs...")
        
        # Exhaustive pairwise comparison (like BERT code)
        for i in tqdm(range(len(addresses_df)), desc="Processing addresses"):
            if count >= max_pairs:
                break
                
            row_a = addresses_df.iloc[i]
            
            for j in range(i + 1, len(addresses_df)):
                if count >= max_pairs:
                    break
                    
                row_b = addresses_df.iloc[j]
                
                # Apply Haversine distance filtering (like BERT code)
                try:
                    lat_a, lon_a = float(row_a['lat']), float(row_a['long'])
                    lat_b, lon_b = float(row_b['lat']), float(row_b['long'])
                    
                    if pd.isna(lat_a) or pd.isna(lon_a) or pd.isna(lat_b) or pd.isna(lon_b):
                        skipped_by_distance += 1
                        continue
                        
                    dist = haversine_distance_m(lat_a, lon_a, lat_b, lon_b)
                    
                    if dist > distance_threshold_m:
                        skipped_by_distance += 1
                        continue
                        
                except (ValueError, TypeError):
                    skipped_by_distance += 1
                    continue
                
                # Create pair for evaluation
                pair_key = frozenset((row_a['orig_index'], row_b['orig_index']))
                true_label = 1 if pair_key in dup_set else 0
                
                # Get model prediction
                try:
                    prediction, generated_text = self.predict_single_pair(
                        str(row_a['text']).strip(), 
                        str(row_b['text']).strip()
                    )
                    
                    # Store results
                    result = {
                        'orig_indexA': row_a['orig_index'],
                        'orig_indexB': row_b['orig_index'],
                        'text_a': str(row_a['text']).strip(),
                        'text_b': str(row_b['text']).strip(),
                        'lat_a': lat_a,
                        'long_a': lon_a,
                        'lat_b': lat_b,
                        'long_b': lon_b,
                        'distance_m': dist,
                        'prediction': prediction,
                        'true_label': true_label,
                        'generated_text': generated_text,
                        'correct': prediction == true_label
                    }
                    
                    results.append(result)
                    predictions.append(prediction)
                    true_labels.append(true_label)
                    
                    # Track wrong predictions (like BERT code)
                    if prediction != true_label:
                        wrong_records.append({
                            'indexA': row_a['orig_index'],
                            'indexB': row_b['orig_index'],
                            'text_a': str(row_a['text']).strip(),
                            'text_b': str(row_b['text']).strip(),
                            'distance_m': dist,
                            'true_label': true_label,
                            'pred_label': prediction,
                            'generated_text': generated_text,
                            'error_type': 'False Positive' if (prediction == 1 and true_label == 0) else 'False Negative'
                        })
                    
                    count += 1
                    processed_pairs += 1
                    
                    # Clear memory periodically
                    if processed_pairs % batch_size == 0:
                        clear_memory()
                        
                        # Print progress
                        if processed_pairs % (batch_size * 10) == 0:
                            current_accuracy = accuracy_score(true_labels, predictions) if len(predictions) > 0 else 0
                            print(f"Processed {processed_pairs} pairs, current accuracy: {current_accuracy:.4f}")
                    
                except Exception as e:
                    print(f"Error processing pair ({row_a['orig_index']}, {row_b['orig_index']}): {e}")
                    continue
        
        print(f"\n" + "="*60)
        print(f"EXHAUSTIVE EVALUATION COMPLETED")
        print(f"="*60)
        print(f"Total candidate pairs generated: {processed_pairs + skipped_by_distance}")
        print(f"Pairs evaluated (within {distance_threshold_m}m): {processed_pairs}")
        print(f"Pairs skipped by distance filter: {skipped_by_distance}")
        print(f"True positives found: {sum(true_labels)}")
        print(f"True negatives found: {len(true_labels) - sum(true_labels)}")
        
        # Calculate metrics
        if len(predictions) > 0:
            metrics = self.calculate_metrics(true_labels, predictions)
            
            print(f"\nMETRICS:")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision_binary']:.4f}")
            print(f"Recall:    {metrics['recall_binary']:.4f}")
            print(f"F1 Score:  {metrics['f1_binary']:.4f}")
            
            print(f"\nCONFUSION MATRIX:")
            try:
                cm = metrics['confusion_matrix']
                print("Predicted:")
                print("         0    1")
                print(f"True 0: {cm[0][0]:4d} {cm[0][1]:4d}")
                print(f"True 1: {cm[1][0]:4d} {cm[1][1]:4d}")
            except Exception:
                print("Could not display confusion matrix")
            
            # Save results
            if save_predictions:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save all results
                results_df = pd.DataFrame(results)
                results_file = f"exhaustive_evaluation_results_{timestamp}.csv"
                results_df.to_csv(results_file, index=False)
                print(f"\nDetailed results saved to: {results_file}")
                
                # Save wrong predictions
                if wrong_records:
                    wrong_df = pd.DataFrame(wrong_records)
                    wrong_file = f"exhaustive_wrong_predictions_{timestamp}.csv"
                    wrong_df.to_csv(wrong_file, index=False)
                    print(f"Wrong predictions saved to: {wrong_file}")
                    print(f"  - False Positives: {len(wrong_df[wrong_df['error_type'] == 'False Positive'])}")
                    print(f"  - False Negatives: {len(wrong_df[wrong_df['error_type'] == 'False Negative'])}")
                
                # Save summary
                summary = {
                    'evaluation_timestamp': timestamp,
                    'total_addresses': len(addresses_df),
                    'total_candidate_pairs': processed_pairs + skipped_by_distance,
                    'pairs_evaluated': processed_pairs,
                    'pairs_skipped_by_distance': skipped_by_distance,
                    'distance_threshold_m': distance_threshold_m,
                    'ground_truth_positives': len(dup_set),
                    'found_positives': sum(true_labels),
                    'evaluation_method': 'exhaustive_pairwise',
                    'metrics': metrics,
                    'model_path': self.model_path,
                    'base_model': self.base_model_name
                }
                
                summary_file = f"exhaustive_evaluation_summary_{timestamp}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                print(f"Summary saved to: {summary_file}")
            
            return pd.DataFrame(results), metrics
        else:
            print("No valid predictions generated!")
            return None, None
    
    def calculate_metrics(self, true_labels, predictions):
        """Calculate comprehensive evaluation metrics"""
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Binary metrics (for the positive class)
        precision_binary = precision_score(true_labels, predictions, zero_division=0)
        recall_binary = recall_score(true_labels, predictions, zero_division=0)
        f1_binary = f1_score(true_labels, predictions, zero_division=0)
        
        conf_matrix = confusion_matrix(true_labels, predictions)
        class_report = classification_report(true_labels, predictions, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_binary': precision_binary,
            'recall_binary': recall_binary,
            'f1_binary': f1_binary,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }

def main():
    """Main evaluation function with exhaustive pairwise comparison"""
    print("="*60)
    print("LLAMA EXHAUSTIVE PAIRWISE ADDRESS MATCHING EVALUATION")
    print("="*60)
    
    # Load configuration
    cfg = load_config()
    
    # File paths from config
    addresses_file = cfg['data']['addresses_csv']
    inference_labels_file = cfg['data']['inference_labels']
    model_path = cfg['training']['output_dir']
    base_model_name = cfg['model']['name_or_path']
    
    # Get parameters from config (like BERT code)
    distance_threshold = float(cfg['data'].get('distance_threshold_m', 600))
    max_pairs = cfg['inference'].get('max_pairs', float('inf'))
    batch_size = cfg['inference'].get('batch_size', 50)
    
    print(f"Addresses file: {addresses_file}")
    print(f"Inference labels file: {inference_labels_file}")
    print(f"Model path: {model_path}")
    print(f"Base model: {base_model_name}")
    print(f"Distance threshold: {distance_threshold}m")
    print(f"Max pairs: {max_pairs}")
    
    # Load data
    print("\nLoading data...")
    try:
        addresses_df = pd.read_csv(addresses_file)
        print(f"Loaded {len(addresses_df)} addresses")
        print(f"Address columns: {addresses_df.columns.tolist()}")
        
        inference_pairs_df = pd.read_csv(inference_labels_file)
        print(f"Loaded {len(inference_pairs_df)} ground truth pairs")
        print(f"Inference columns: {inference_pairs_df.columns.tolist()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Validate data
    required_addr_cols = ['orig_index', 'text', 'lat', 'long']
    required_inf_cols = ['orig_indexA', 'orig_indexB']
    
    if not all(col in addresses_df.columns for col in required_addr_cols):
        print(f"Error: addresses.csv missing required columns: {required_addr_cols}")
        return
    
    if not all(col in inference_pairs_df.columns for col in required_inf_cols):
        print(f"Error: inference_labels.csv missing required columns: {required_inf_cols}")
        return
    
    print("✓ Data validation passed")
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    try:
        evaluator = AddressEvaluator(
            model_path=model_path,
            base_model_name=base_model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run exhaustive evaluation
    print("\nStarting exhaustive pairwise evaluation...")
    results_df, metrics = evaluator.evaluate_exhaustive_pairs(
        addresses_df=addresses_df,
        inference_pairs_df=inference_pairs_df,
        distance_threshold_m=distance_threshold,
        max_pairs=max_pairs,
        save_predictions=True,
        batch_size=batch_size
    )
    
    if results_df is not None:
        print("\n✓ Exhaustive evaluation completed successfully!")
        
        # Show sample results
        print(f"\nSample Results ({len(results_df)} total pairs):")
        print("-" * 80)
        sample_results = results_df.head(5)
        for _, row in sample_results.iterrows():
            status = "✓" if row['correct'] else "✗"
            print(f"{status} Pair {row['orig_indexA']}-{row['orig_indexB']} (dist: {row['distance_m']:.0f}m)")
            print(f"  A: {row['text_a'][:50]}...")
            print(f"  B: {row['text_b'][:50]}...")
            print(f"  Pred: {row['prediction']} | True: {row['true_label']} | {row['generated_text']}")
            print()
    else:
        print("\n❌ Exhaustive evaluation failed!")

if __name__ == "__main__":
    main()




