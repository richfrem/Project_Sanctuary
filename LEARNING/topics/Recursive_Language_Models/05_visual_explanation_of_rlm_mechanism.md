# Visual Explanation: The "Prompt as Environment" Mechanism

**User Query:** "Explain 'treats long prompts as part of an external environment... programmatically examine...'"

## 1. The Core Shift: "Inside" vs "Outside"

To understand RLM, you must visualize where the "Prompt" lives.

### A. The Standard Way (Prompt is INSIDE)
The prompt is fed directly into the LLM's "Attention" (Brain RAM).
*   **Visual:** `LLM( [The ENTIRE 10MB Novel] )`
*   **Problem:** The LLM's brain is full. It gets confused. It costs a fortune in compute to "attend" to every word at once.

### B. The RLM Way (Prompt is OUTSIDE)
The prompt is stored in a **Python Variable** on a server. The LLM never sees the whole thing. It only sees a "Pointer" to it.
*   **Visual:** `LLM( "There is a variable called 'BOOK' loaded in your environment. It has 10 million characters. What do you want to do?" )`
*   **The "Environment":** A standard Python REPL (Read-Eval-Print Loop).

## 2. "Programmatically Examine" (The Magnifying Glass)
Since the LLM can't see the text, it has to use **Code** to see it. It acts like a blind programmer navigating a file.

**LLM Thinking:** "I need to check the beginning of the book to see who the main character is."
**LLM Action (Code):**
```python
# The LLM writes this code to "peek"
print(BOOK[:1000])  # Read first 1000 chars
```
**Environment Output:** "It was the best of times, it was the worst of times..."
**LLM Result:** "Okay, I see the text now."

## 3. "Decompose" (Slicing the Cake)
The LLM realizes the book is too big to read at once. It writes code to chop it up.

**LLM Action (Code):**
```python
# The LLM calculates chunk sizes
total_len = len(BOOK)
chunk_size = 5000
chunks = [BOOK[i : i + chunk_size] for i in range(0, total_len, chunk_size)]
```

## 4. "Recursively Call Itself" (Spawning Sub-Agents)
This is the magic step. The LLM creates *copies* of itself to do the heavy lifting.

**LLM Action (Code):**
```python
narrative_arcs = []

for chunk in chunks:
    # RECURSION: The LLM calls the 'llm_query' function
    # This spawns a FRESH, EMPTY LLM that only sees this tiny chunk
    summary = llm_query(
        prompt="Summarize the plot events in this text snippet.",
        context=chunk  # Only passing 5,000 chars, not 10 million!
    )
    narrative_arcs.append(summary)

# Aggregation
final_summary = "\n".join(narrative_arcs)
```

## Summary of the Mechanism
1.  **Externalize:** The text sits in RAM, not in the Neural Network.
2.  **Examine:** The Network uses Python functions (`len`, `slice`) to "touch" the data.
3.  **Recurse:** The Network outsources the reading. It acts as a **Manager**, shrinking the task into 100 small jobs, assigning them to 100 "Sub-Agents" (recursive calls), and then compiling the report.
