#!/usr/bin/env python3
"""
validate_inquiry.py
Basic CLI validator for inquiry JSON against 87_inquiry_schema.json
Usage:
  python tools/steward_validation/validate_inquiry.py mnemonic_cortex/INQUIRY_TEMPLATES/samples/sample_queries.json
"""
import json, sys, os
from jsonschema import validate, ValidationError

SCHEMA_PATH = os.path.join('mnemonic_cortex','INQUIRY_TEMPLATES','87_inquiry_schema.json')

def load(path):
    with open(path,'r',encoding='utf8') as f:
        return json.load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: validate_inquiry.py <queries.json>")
        sys.exit(2)
    queries = load(sys.argv[1])
    schema = load(SCHEMA_PATH)
    ok = True
    for i,q in enumerate(queries):
        try:
            validate(instance=q, schema=schema)
            print(f"[OK ] query #{i} request_id={q.get('request_id')}")
        except ValidationError as e:
            ok = False
            print(f"[ERR] query #{i} request_id={q.get('request_id')} -> {e.message}")
    sys.exit(0 if ok else 3)

if __name__ == "__main__":
    main()