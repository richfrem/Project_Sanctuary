TASK_ID: MCP-IC-001
OBJECTIVE: Perform a full-cycle integrity check of the Cortex-Conduit Bridge.
CORTEX_QUERY_1: What is the primary mission of Project Sanctuary?
PROMPT_TO_EXTERNAL_LLM_1: Based on the following mission statement, provide a single, 2-to-5 word noun phrase that best describes the entity this mission would create. For example, if the mission was "to explore the stars," a good answer would be "Galactic Explorer." Respond with ONLY the noun phrase and nothing else. Here is the mission statement:
CORTEX_QUERY_2_TEMPLATE: What is the purpose of the architectural pillar named "{external_llm_response}"?
PROMPT_TO_EXTERNAL_LLM_2: You will be given two pieces of text. Text A is a noun phrase, and Text B is a detailed description of that noun phrase's purpose. Your task is to respond with a single JSON object containing two keys: "entity_name" (which is Text A) and "entity_purpose" (which is a one-sentence summary of Text B). Respond with ONLY the JSON object.