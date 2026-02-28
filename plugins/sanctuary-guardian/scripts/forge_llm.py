#!/usr/bin/env python3
"""
forge_llm.py — Sanctuary Fine-Tuned Model Interface
=====================================================

Purpose:
    CLI interface for querying the fine-tuned Sanctuary model via Ollama.
    Standalone script — no mcp_servers dependencies.

Usage:
    python3 forge_llm.py query "What are the core principles of Project Sanctuary?"
    python3 forge_llm.py query "Explain Protocol 128" --temperature 0.5
    python3 forge_llm.py status

Layer: Plugin Script (guardian-onboarding)

Dependencies:
    - ollama (pip install ollama)
"""

import os
import sys
import json
import argparse


SANCTUARY_MODEL = "hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M"


def query_model(prompt: str, temperature: float = 0.7, max_tokens: int = 2048,
                system_prompt: str = None) -> dict:
    """
    Query the fine-tuned Sanctuary model via Ollama.
    Returns a JSON-serializable dict with the response.
    """
    try:
        from ollama import Client

        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        client = Client(host=ollama_host)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat(
            model=SANCTUARY_MODEL,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        answer = response['message']['content']
        prompt_tokens = response.get('prompt_eval_count')
        completion_tokens = response.get('eval_count')
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens and completion_tokens else None

        return {
            "status": "success",
            "model": SANCTUARY_MODEL,
            "response": answer,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "temperature": temperature
        }

    except ImportError:
        return {
            "status": "error",
            "error": "ollama package not installed. Install with: pip install ollama"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to query model: {str(e)}"
        }


def check_status() -> dict:
    """
    Check if the Sanctuary model is available in Ollama.
    """
    try:
        import ollama

        models_response = ollama.list()

        if isinstance(models_response, dict):
            models_list = models_response.get('models', [])
        else:
            models_list = models_response

        model_names = [
            m.get('name', m.get('model', str(m))) if isinstance(m, dict) else str(m)
            for m in models_list
        ]

        is_available = any(SANCTUARY_MODEL in name for name in model_names)

        return {
            "status": "success",
            "model": SANCTUARY_MODEL,
            "available": is_available,
            "all_models": model_names
        }

    except ImportError:
        return {"status": "error", "error": "ollama package not installed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Sanctuary Fine-Tuned Model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # query
    q_parser = subparsers.add_parser("query", help="Query the Sanctuary model")
    q_parser.add_argument("prompt", help="Prompt to send to the model")
    q_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    q_parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    q_parser.add_argument("--system", help="System prompt for context")

    # status
    subparsers.add_parser("status", help="Check model availability")

    args = parser.parse_args()

    if args.command == "query":
        result = query_model(
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=getattr(args, 'max_tokens', 2048),
            system_prompt=getattr(args, 'system', None)
        )
        if result["status"] == "success":
            print(f"\n{result['response']}")
            print(f"\n📊 Tokens: {result.get('total_tokens', 'N/A')} | Temp: {result['temperature']}")
        else:
            print(json.dumps(result, indent=2))
            sys.exit(1)

    elif args.command == "status":
        result = check_status()
        print(json.dumps(result, indent=2))
        if result.get("status") != "success":
            sys.exit(1)


if __name__ == "__main__":
    main()
