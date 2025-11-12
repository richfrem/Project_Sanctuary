#!/bin/bash

# A simple smoke test harness for verifying the different cognitive layers of the Mnemonic Cortex.
# This script must be run from the project root directory.

echo "--- STARTING COGNITIVE LAYER VERIFICATION (v3 - Final) ---"

# Test 1: Internal Model Memory (Fine-Tune Only)
# Goal: Verify the model can answer from its foundational, weeks-old trained knowledge using Protocol 63 as a Temporal Anchor.
echo -e "\n--- TEST 1: Internal Model Memory (--no-rag) ---"
echo "Querying for: 'What are the three core cognitive roles in Protocol 63?'"
python3 mnemonic_cortex/app/main.py --no-rag "What are the three core cognitive roles in Protocol 63?" | grep -E -i 'synthesizer|red teamer|translator'

if [ $? -eq 0 ]; then
    echo "  [+] SUCCESS: Internal model memory test passed. Found key concept from foundational Protocol 63."
else
    echo "  [-] FAILURE: Internal model memory test failed. Did NOT find key concept from foundational Protocol 63."
fi

# Test 2: Retrieval Integrity (RAG-Only)
# Goal: Verify the RAG pipeline retrieves the correct document from the database.
echo -e "\n--- TEST 2: RAG Retrieval Integrity (--retrieve-only) ---"
echo "Querying for: 'the doctrine of unbreakable git commits'"
python3 mnemonic_cortex/app/main.py --retrieve-only "the doctrine of unbreakable git commits" | grep "101_The_Doctrine_of_the_Unbreakable_Commit.md"

if [ $? -eq 0 ]; then
    echo "  [+] SUCCESS: RAG retrieval test passed. Correct protocol document was retrieved."
else
    echo "  [-] FAILURE: RAG retrieval test failed. Correct protocol document was NOT retrieved."
fi

# Test 3: Agentic Loop (End-to-End)
# Goal: Verify the full cognitive loop from high-level goal to final, RAG-augmented answer.
echo -e "\n--- TEST 3: Full Agentic Loop (agentic_query.py) ---"
echo "Querying for high-level goal: 'Explain the doctrine of unbreakable git commits.'"
python3 mnemonic_cortex/scripts/agentic_query.py "Explain the doctrine of unbreakable git commits." | grep -i "P101"

if [ $? -eq 0 ]; then
    echo "  [+] SUCCESS: Full agentic loop test passed. Final answer referenced the correct protocol."
else
    echo "  [-] FAILURE: Full agentic loop test failed. Final answer did NOT reference the correct protocol."
fi

echo -e "\n--- COGNITIVE LAYER VERIFICATION COMPLETE ---"
#!/bin/bash

# A simple smoke test harness for verifying the different cognitive layers of the Mnemonic Cortex.
# This script must be run from the project root directory.

echo "--- STARTING COGNITIVE LAYER VERIFICATION ---"

# Test 1: Internal Model Knowledge (Fine-Tune Only)
# Goal: Verify the model can answer from its trained knowledge without retrieval.
echo -e "\n--- TEST 1: Internal Model Memory (--no-rag) ---"
echo "Querying for: 'What is the core principle of the Anvil Protocol?'"
python3 mnemonic_cortex/app/main.py --no-rag "What is the core principle of the Anvil Protocol?" | grep -i "forged, tested, documented"

if [ $? -eq 0 ]; then
    echo "  [+] SUCCESS: Internal model memory test passed. Expected phrase found."
else
    echo "  [-] FAILURE: Internal model memory test failed. Expected phrase NOT found."
fi

# Test 2: Retrieval Integrity (RAG-Only)
# Goal: Verify the RAG pipeline retrieves the correct document from the database.
echo -e "\n--- TEST 2: RAG Retrieval Integrity (--retrieve-only) ---"
echo "Querying for: 'the doctrine of unbreakable git commits'"
python3 mnemonic_cortex/app/main.py --retrieve-only "the doctrine of unbreakable git commits" | grep "101_The_Doctrine_of_the_Unbreakable_Commit.md"

if [ $? -eq 0 ]; then
    echo "  [+] SUCCESS: RAG retrieval test passed. Correct protocol document was retrieved."
else
    echo "  [-] FAILURE: RAG retrieval test failed. Correct protocol document was NOT retrieved."
fi

# Test 3: Agentic Loop (End-to-End)
# Goal: Verify the full cognitive loop from high-level goal to final, RAG-augmented answer.
echo -e "\n--- TEST 3: Full Agentic Loop (agentic_query.py) ---"
echo "Querying for high-level goal: 'Explain the doctrine of unbreakable git commits.'"
python3 mnemonic_cortex/scripts/agentic_query.py "Explain the doctrine of unbreakable git commits." | grep -i "P101"

if [ $? -eq 0 ]; then
    echo "  [+] SUCCESS: Full agentic loop test passed. Final answer referenced the correct protocol."
else
    echo "  [-] FAILURE: Full agentic loop test failed. Final answer did NOT reference the correct protocol."
fi

echo -e "\n--- COGNITIVE LAYER VERIFICATION COMPLETE ---"
