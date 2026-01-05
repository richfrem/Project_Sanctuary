#!/usr/bin/env python3
#============================================
# forge/tests/verify_adapter.py
# Purpose: Verify fine-tuned LoRA adapters by running test inference.
# Role: Verification Layer
# Used by: Post-training verification, adapter validation
#============================================
"""
Inference Test Script for Project Sanctuary Sovereign AI Models.

This script runs inference using the fine-tuned Project Sanctuary model.
It supports loading either the LoRA adapter (post-fine-tune) or the merged model (post-merge).
Uses 4-bit quantization for compatibility with 8GB GPUs.

Usage:
    # Test adapter after fine-tune
    python forge/tests/verify_adapter.py --input "What is the Doctrine of Sovereign Resilience?"

    # Test merged model after merge
    python forge/tests/verify_adapter.py --model-type merged --input "Test prompt"

    # Force GPU loading with 4-bit quantization
    python forge/tests/verify_adapter.py --device cuda --load-in-4bit --input "Test prompt"

References:
    - ADR 075: Standardized Code Documentation Pattern
    - Protocol 41: Operation Phoenix Forge
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Project Utilities Bootstrap ---
SCRIPT_DIR = Path(__file__).resolve().parent  # forge/tests
FORGE_ROOT = SCRIPT_DIR.parent                 # forge
PROJECT_ROOT = FORGE_ROOT.parent               # Project_Sanctuary

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mcp_servers.lib.path_utils import find_project_root
    from mcp_servers.lib.logging_utils import setup_mcp_logging
    PROJECT_ROOT = Path(find_project_root())
    logger = setup_mcp_logging("sanctuary.inference", log_file="logs/inference.log")
except ImportError:
    # Fallback for WSL environment
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("sanctuary.inference")

# --- Load environment variables from project root .env ---
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed, rely on system environment

# --- Configuration Constants ---
CONFIG_PATH = FORGE_ROOT / "config" / "inference_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- Environment Variable Fallbacks ---
# Model paths
DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / os.environ.get('SANCTUARY_BASE_MODEL_PATH', config["model"]["base_path"])
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / os.environ.get('SANCTUARY_ADAPTER_PATH', config["model"]["adapter_path"])
DEFAULT_MERGED_PATH = PROJECT_ROOT / os.environ.get('SANCTUARY_MERGED_MODEL_PATH', config["model"]["merged_path"])

# 4-bit quantization config for 8GB GPU compatibility
quant_config = config["quantization"]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=os.environ.get('SANCTUARY_LOAD_IN_4BIT', str(quant_config["load_in_4bit"])).lower() == 'true',
    bnb_4bit_compute_dtype=torch.bfloat16 if os.environ.get('SANCTUARY_BNB_4BIT_COMPUTE_DTYPE', quant_config["bnb_4bit_compute_dtype"]) == "bfloat16" else torch.float16,
    bnb_4bit_use_double_quant=os.environ.get('SANCTUARY_BNB_4BIT_USE_DOUBLE_QUANT', str(quant_config["bnb_4bit_use_double_quant"])).lower() == 'true',
    bnb_4bit_quant_type=os.environ.get('SANCTUARY_BNB_4BIT_QUANT_TYPE', quant_config["bnb_4bit_quant_type"])
)

def load_model_and_tokenizer(model_type, device="auto", load_in_4bit=None):
    """Loads the model and tokenizer based on type (adapter or merged)."""
    # Override quantization if specified
    if load_in_4bit is not None:
        bnb_config.load_in_4bit = load_in_4bit
    
    # Force device_map for CUDA to pin to GPU 0
    if device == "cuda":
        device_map = "cuda:0"
    else:
        device_map = device
    
    if model_type == "adapter":
        base_path = DEFAULT_BASE_MODEL_PATH
        adapter_path = DEFAULT_ADAPTER_PATH
        if not base_path.exists():
            print(f"🛑 CRITICAL FAILURE: Base model not found at '{base_path}'.")
            print("Please run the download script first.")
            sys.exit(1)
        if not adapter_path.exists():
            print(f"🛑 CRITICAL FAILURE: Adapter not found at '{adapter_path}'.")
            print("Please run fine_tune.py first.")
            sys.exit(1)
        
        print(f"🧠 Loading base model from: {base_path}...")
        print(f"🔧 Using device_map: {device_map}")
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Optimize memory usage
        )
        print(f"🔧 Applying adapter from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    
    elif model_type == "merged":
        model_path = DEFAULT_MERGED_PATH
        if not model_path.exists():
            print(f"🛑 CRITICAL FAILURE: Merged model not found at '{model_path}'.")
            print("Please run merge_adapter.py first.")
            sys.exit(1)
        
        print(f"🧠 Loading merged model from: {model_path}...")
        print(f"🔧 Using device_map: {device_map}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Optimize memory usage
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    else:
        print(f"🛑 ERROR: Invalid model_type '{model_type}'. Use 'adapter' or 'merged'.")
        sys.exit(1)
    
    print("✅ Model and tokenizer loaded successfully.")
    return model, tokenizer

def format_prompt(instruction):
    """Formats the user's question into the Qwen2 ChatML format."""
    # The system prompt is implicitly handled by the fine-tuned model's training.
    # We only need to provide the user's query.
    prompt = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

def run_inference(model, tokenizer, instruction, max_new_tokens, temperature=None, top_p=None, do_sample=None):
    """Generates a response from the model for a given instruction."""
    prompt = format_prompt(instruction)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get generation config with environment variable fallbacks
    gen_config = config["generation"]
    
    # Use provided args or fall back to config/env
    if temperature is None:
        temperature = float(os.environ.get('SANCTUARY_TEMPERATURE', gen_config["temperature"]))
    if top_p is None:
        top_p = float(os.environ.get('SANCTUARY_TOP_P', gen_config["top_p"]))
    if do_sample is None:
        do_sample = os.environ.get('SANCTUARY_DO_SAMPLE', str(gen_config["do_sample"])).lower() == 'true'
    
    # Generate the response
    with torch.no_grad():
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample,
            'pad_token_id': tokenizer.eos_token_id
        }
        if do_sample:
            gen_kwargs['top_k'] = 50  # Add for stability
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode and return only the generated part of the response
    response_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned Project Sanctuary model.")
    parser.add_argument('--model-type', choices=['adapter', 'merged'], default='adapter', 
                        help='Type of model to load: adapter (post-fine-tune) or merged (post-merge).')
    parser.add_argument('--input', help='A direct question or instruction to ask the model.')
    parser.add_argument('--file', help='Path to a file to use as the input instruction.')
    parser.add_argument('--max-new-tokens', type=int, default=int(os.environ.get('SANCTUARY_MAX_NEW_TOKENS', config["generation"]["max_new_tokens"])), help='Maximum number of new tokens to generate.')
    
    # GPU and quantization options
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda', 
                        help='Device mapping for model loading. Use "cuda" to force GPU.')
    parser.add_argument('--load-in-4bit', action='store_true', default=None,
                        help='Enable 4-bit quantization (overrides config).')
    parser.add_argument('--no-load-in-4bit', action='store_true', 
                        help='Disable 4-bit quantization (overrides config).')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, 
                        help='Sampling temperature (0.0 = greedy, higher = more random).')
    parser.add_argument('--top-p', type=float, 
                        help='Top-p nucleus sampling parameter.')
    parser.add_argument('--do-sample', action='store_true', default=None,
                        help='Enable sampling (required for temperature/top-p to take effect).')
    parser.add_argument('--greedy', action='store_true', 
                        help='Force greedy decoding (do_sample=False, temperature=0).')
    
    args = parser.parse_args()
    
    # Load YAML if exists and override args
    config_path = Path(__file__).parent.parent / "config" / "inference_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Override args with YAML if not provided
        if yaml_config.get('quantization', {}).get('load_in_4bit') and args.load_in_4bit is None:
            args.load_in_4bit = True
        if yaml_config.get('generation', {}).get('max_new_tokens') and args.max_new_tokens == int(os.environ.get('SANCTUARY_MAX_NEW_TOKENS', config["generation"]["max_new_tokens"])):
            args.max_new_tokens = yaml_config['generation']['max_new_tokens']
        if yaml_config.get('generation', {}).get('temperature') and args.temperature is None:
            args.temperature = yaml_config['generation']['temperature']
        if yaml_config.get('generation', {}).get('top_p') and args.top_p is None:
            args.top_p = yaml_config['generation']['top_p']
        if yaml_config.get('generation', {}).get('do_sample') is not None and args.do_sample is None:
            args.do_sample = yaml_config['generation']['do_sample']
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to 'auto'")
        args.device = 'auto'
    
    # Resolve quantization flag
    load_in_4bit = None
    if args.load_in_4bit:
        load_in_4bit = True
    elif args.no_load_in_4bit:
        load_in_4bit = False
    
    # Resolve generation flags
    do_sample = args.do_sample
    if args.greedy:
        do_sample = False
        args.temperature = 0.0
    
    print(f"🔧 Loading with device_map='{args.device}', 4-bit quantization={load_in_4bit if load_in_4bit is not None else 'config default'}")
    print(f"🔧 Generation: do_sample={do_sample}, temperature={args.temperature or 'config default'}, top_p={args.top_p or 'config default'}")
    
    model, tokenizer = load_model_and_tokenizer(args.model_type, device=args.device, load_in_4bit=load_in_4bit)

    instruction_text = ""
    source_name = ""

    if args.input:
        instruction_text = args.input
        source_name = "direct input"
    elif args.file:
        try:
            source_path = Path(args.file)
            instruction_text = source_path.read_text(encoding='utf-8')
            source_name = f"file: {source_path.name}"
        except FileNotFoundError:
            print(f"🛑 ERROR: Input file not found at '{args.file}'")
            sys.exit(1)
    else:
        print("▶️  No input provided via --input or --file. Reading from stdin...")
        print("▶️  Enter your instruction below, then press Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows) to run.")
        instruction_text = sys.stdin.read()
        source_name = "stdin"

    print(f"\n---  querying model based on {source_name} ---")
    
    response = run_inference(model, tokenizer, instruction_text, args.max_new_tokens, 
                           temperature=args.temperature, top_p=args.top_p, do_sample=do_sample)
    
    print("\n" + "="*80)
    print("✅ Sovereign AI Response:")
    print("="*80)
    print(response)
    print("="*80)

if __name__ == "__main__":
    main()