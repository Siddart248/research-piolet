# Tool and Function Calling in LLMs

Tool calling (or function calling) enables Large Language Models to interface with deterministic software systems, external APIs, databases, and computational utilities.

## Mechanics of Function Calling

Large Language Models do not execute code or access external networks directly in their forward pass. Instead, tool calling operates via a contract:
1. **Schema Declaration**: The host application provides function definitions, parameter names, types, descriptions, and required fields specified via JSON Schema or Pydantic models.
2. **Model Invocation with Tools**: When prompted, the model determines whether a registered tool is needed. If needed, rather than producing conversational text, the model generates a structured payload specifying `function_name` and `arguments`.
3. **Host Execution**: The host application intercepts the structured call, validates the parameters, executes the actual code (e.g., executing a SQL query or calculating a formula), and captures the output.
4. **Observation Injection**: The tool's return value is appended to the prompt history as a tool response message. The LLM then consumes this ground-truth data to formulate its answer.

## Advantages Over Raw Text Parsing

Early approaches attempted to extract commands using regex from unstructured text (e.g., "ACTION: run_query(...)"). Native tool calling provides:
- **Constrained Decoding**: Grammars ensure emitted arguments conform strictly to the declared JSON schema.
- **Strong Typing**: Eliminates syntactic parsing errors and parameter hallucination.
- **Parallel Tool Calling**: Many modern models can emit multiple independent tool calls in a single turn.

## Common Tool Calling Pitfalls

- **Over-invocation**: The model invokes tools for trivial queries that could be handled directly from instructions.
- **Argument Hallucination**: Passing parameters not permitted by the schema or hallucinating entity IDs.
- **Unhandled Exceptions**: Crashing the workflow when a tool returns an error instead of passing the error back into context for recovery.
