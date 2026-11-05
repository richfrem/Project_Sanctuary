# The Synthesis Phase: How RLM Reassembles the Pieces

**User Query:** "How does the article propose reassembling / synthesizing all the recursive bits?"

The article proposes **two primary methods** for reassembly, depending on the complexity of the task. They both fundamentally rely on the Root Agent (Manager) having access to the *outputs* of the sub-agents (but not their full context).

## Method A: The Linear Accumulator (Loop & Buffer)
*Best for: Summarization, Narrative Extraction*

1.  **The Loop:** The Agent iterates through chunks.
2.  **The Sub-Call:** `summary = llm_query(chunk)`
3.  **The Accumulation:** The Agent appends this `summary` to a list or string variable in the Python environment (e.g., `chapter_summaries`).
4.  **The Final Context:** When the loop finishes, the `chapter_summaries` list (which might be 2,000 tokens) *becomes the context* for the final query.
5.  **The Final Call:** `final_answer = llm_query("Based on these summaries... what is the conclusion?", context=chapter_summaries)`

**Analogy:** A manager reads 10 weekly reports from subordinates, pastes them into one document, and writes a Monthly Executive Summary.

## Method B: The Programmatic Aggregation (Code Logic)
*Best for: Exact Counting, Filtering (OOLONG Benchmark)*

1.  **The Loop:** The Agent iterates through chunks.
2.  **The Sub-Call:** `result = llm_query("Extract all user IDs and their timestamps from this chunk.")`
3.  **The Logic:** The Agent *does not* just paste the text. It uses Python code to parse the result.
    *   *Example:* `data = json.loads(result)`
    *   *Logic:* `all_users.extend(data['users'])`
4.  **The Synthesis:** The final answer isn't an LLM summary; it's the result of the code execution.
    *   *Example:* `final_answer = len(set(all_users))`
    *   *Or:* `final_answer = sort(all_users)`
5.  **The Output:** The Agent returns the computed value.

**Analogy:** A census bureau collects spreadsheets from 50 states. It doesn't write a poem about them; it sums the "Population" column to get a final number.

## Key Insight: "Variables as Bridge"
The "Context" for the final answer is **whatever data structures (lists, dicts, strings)** the Root Agent built during its recursion loop.
*   It explicitly *discards* the raw chunks (saving memory).
*   It *keeps* the distilled insights (in variables).
*   The final synthesis acts only on those distilled variables.
