```markdown
# Documentation for Decathlon Copy Generation Agent

This documentation provides a detailed overview of the Decathlon Copy Generation Agent, including its architecture, components, and usage.

## Overview

The Decathlon Copy Generation Agent is designed to automate the creation of marketing copy for Decathlon CRM emails. It utilizes a state-based graph workflow built with Langchain and LangGraph, leveraging the capabilities of OpenAI's language models. The agent takes predefined content components as input, generates corresponding text, validates the generated content against specific criteria, and finally formats the output into a structured JSON file.

## Architecture

The agent's architecture is based on a LangGraph `StateGraph`, which orchestrates the following key nodes:

1. **`generate`:** This node is responsible for generating content for each `CopyComponent`. It constructs prompts based on component specifications, knowledge base information, and feedback from previous attempts. It uses the OpenAI `ChatOpenAI` model for content generation with retry logic for robustness.

2. **`validate`:** This node validates the generated content against predefined rules. These rules include checking character limits, ensuring the content is not empty, and verifying the absence of unwanted prefixes. Feedback is generated based on validation failures.

3. **`format_output`:** This node formats the final output into a clean JSON structure. It includes the generated content, validation status, and any error messages for each component. The output is then saved to a file in the `exports` directory.

The transitions between these nodes are managed by conditional edges based on the `should_continue` function, which determines whether to proceed to formatting, retry generation, or terminate the workflow.

## Key Components

### `CopyComponent`

This `dataclass` defines the structure for specifying the requirements of a content element.

```python
@dataclass
class CopyComponent:
    """
    Represents a content component for copy generation.

    Attributes:
        component_type (str): Type of the component (e.g., 'headline basic').
        element_type (str): Specific element within the component type (e.g., 'headline_text').
        char_limit (int): Maximum number of characters allowed for the content.
        token_limit (int): Maximum number of tokens allowed for the content.
        audience (str): Target audience for the content.
        max_attempts (int): Maximum number of generation attempts allowed for this component. Defaults to 3.
    """
    component_type: str
    element_type: str
    char_limit: int
    token_limit: int
    audience: str
    max_attempts: int = 3
```

### `State`

This `TypedDict` defines the state object that is passed between the nodes in the LangGraph workflow.

```python
class State(TypedDict):
    """
    Represents the state of the copy generation workflow.

    Attributes:
        components (List[CopyComponent]): A list of copy components to be generated.
        generated_content (List[Dict[str, Any]]): A list to store the generated content for each component.
        validation_results (List[Dict[str, Any]]): A list to store the validation results for each generated content.
        errors (List[str]): A list to store any error messages encountered during generation or validation.
        attempt_count (int): The current attempt count for generating content.
        status (str): The current status of the workflow (e.g., 'generating', 'validation_passed').
        generation_history (List[Dict[str, Any]]): A list to store the history of generated content for each attempt.
        output (List[Dict[str, Any]]): The final formatted output of the generated content.
    """
    components: List[CopyComponent]
    generated_content: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[Dict[str, Any]]
    output: List[Dict[str, Any]]
```

### Functions

- **`load_knowledge_base(filepath: str) -> dict`:** Loads the knowledge base from a JSON file, providing examples and guidelines for content generation.
- **`generate_with_retry(llm, prompt: str)`:** An asynchronous function that generates content using the provided language model with a retry mechanism.
- **`construct_prompt(component: CopyComponent, kb_info: Dict[str, Any], feedback_text: str) -> str`:** Constructs the prompt for the LLM based on the component details, knowledge base information, and feedback from previous attempts.
- **`generate_feedback(validation_results: Dict[str, Any], component: CopyComponent) -> List[str]`:** Generates human-readable feedback messages based on the validation results.
- **`clean_content(content: str) -> str`:** Cleans the generated content by removing unwanted prefixes and whitespace.
- **`generate_content(state: State) -> State`:**  The core function for generating content within the workflow. It iterates through the components and uses the LLM to generate text.
- **`validate_content(state: State) -> State`:** Validates the generated content against defined criteria.
- **`should_continue(state: State) -> str`:** Determines the next step in the workflow based on the current state.
- **`format_output(state: State) -> State`:** Formats the final output into a JSON structure and saves it to a file.

## Workflow

The workflow operates as follows:

1. **Initialization:** The workflow starts with an initial state containing a list of `CopyComponent` objects.
2. **Generation:** The `generate` node uses the LLM to create content for each component based on the constructed prompts.
3. **Validation:** The `validate` node checks the generated content against predefined rules.
4. **Decision:** The `should_continue` function determines the next step:
   - If validation passes, the workflow proceeds to `format_output`.
   - If validation fails and the maximum attempts have not been reached, the workflow returns to the `generate` node with feedback.
   - If validation fails and the maximum attempts have been reached, the workflow terminates.
5. **Formatting:** The `format_output` node structures the generated content and validation results into a JSON file.
6. **Termination:** The workflow ends, and the final output is saved.

## Getting Started

Refer to the [README.md](README.md) file for detailed instructions on installation, setup, and usage.

## Configuration

The agent's behavior can be configured through:

- **Environment Variables:** The `OPENAI_API_KEY` should be set in a `.env` file.
- **Knowledge Base:** The `template_kb.json` file allows customization of examples and guidelines for content generation.
- **Logging:** The logging level can be adjusted to control the verbosity of the output.

## Output

The generated output is saved as a JSON file in the `exports` directory. The filename includes a timestamp for easy identification. The JSON structure contains the timestamp of generation, the overall status, details of each component (including generated content and validation status), and the generation history.

## Docker

When running in a Docker container, the generated files can be accessed using the `docker cp` command as described in the [README.md](README.md).

## Troubleshooting

- **API Key Issues:** Ensure your OpenAI API key is correctly set in the `.env` file.
- **Validation Failures:** Review the error messages in the output JSON to understand why content validation failed. Adjust the `CopyComponent` parameters or the knowledge base as needed.
- **Unexpected Behavior:** Examine the logs for detailed information about the workflow execution.

## Future Enhancements

- Implement more sophisticated validation rules.
- Allow for human-in-the-loop feedback during the generation process.
- Integrate with other content management systems.
- Improve the prompt engineering for better content quality.
```
