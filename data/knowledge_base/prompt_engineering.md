# Prompt Engineering Principles

Prompt engineering is the empirical practice of designing, structuring, and optimizing textual inputs to guide generative models toward desired behaviors, formats, and reasoning paths without updating model weights.

## Core Prompt Techniques

1. **System Instructions (Personas & Constraints)**:
   Establishing operational boundaries, behavioral tone, role definitions, and strict negative constraints (e.g., "Do not assume facts outside provided context").

2. **Few-Shot In-Context Learning**:
   Providing input-output demonstration pairs within the prompt. Few-shot prompting demonstrates formatting expectations, reasoning styles, and boundary conditions far more effectively than abstract prose instructions.

3. **Chain-of-Thought (CoT) Prompting**:
   Encouraging the model to generate intermediate reasoning steps before arriving at a final answer. Techniques like "Think step-by-step" or explicit multi-step deliberation templates significantly boost performance on arithmetic, logic, and multi-hop reasoning tasks.

4. **Structured Output Enforcement**:
   Prompting for machine-readable formats such as JSON schemas, XML tags, or Markdown tables. Modern models combine prompt-level schema instructions with constrained grammar decoding (like Pydantic integration) to guarantee schema compliance.

## Best Practices and Anti-Patterns

- **Clear Delimiters**: Use delimiters (such as markdown headers, XML tags `<context>...</context>`, or triple backticks) to distinguish instructions from untrusted user inputs or retrieved context documents.
- **Positive Guidance Over Pure Negation**: Describe what the model *should* do rather than solely listing prohibited actions.
- **Avoid Ambiguity**: Specify expected lengths, styles, and criteria for handling missing information directly in the prompt.
