
# Decathlon Copy Generation Agent

This project implements an agent for generating marketing copy for Decathlon CRM emails. It uses Langchain and LangGraph to orchestrate a workflow involving content generation, validation, and formatting.

## Overview

The agent takes a list of `CopyComponent` objects as input, each defining the requirements for a specific piece of content (e.g., headline, body text). It then uses an OpenAI language model to generate content for each component, validates the generated content against predefined rules (like character limits), and formats the final output into a JSON file.

The workflow includes retry mechanisms to handle potential errors during content generation and provides feedback to the language model based on validation failures.

## Features

- **Component-based Content Generation:** Generates content based on predefined components with specific requirements.
- **OpenAI Integration:** Leverages the power of OpenAI's language models for content creation.
- **Validation:** Ensures generated content adheres to specified constraints (character limits, formatting).
- **Retry Logic:** Implements retry mechanisms for robust content generation.
- **Feedback Loop:** Provides feedback to the language model for iterative improvement.
- **Structured Output:** Formats the generated content into a clean JSON structure.
- **Logging:** Provides detailed logging of the generation process.
- **Docker Support:** Instructions for copying generated exports from a Docker container.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root directory of the project.
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
     ```

5. **Prepare the knowledge base:**
   - Ensure you have a `template_kb.json` file in the same directory as the main script. This file contains examples and guidelines for content generation. You can modify this file to suit your specific needs.

## Usage

The main entry point for the agent is the compiled LangGraph workflow. You will need to define the `components` within your application that will be passed to the workflow.

Here's a basic example of how you might define the initial state and invoke the graph:

```python
from your_module import graph, State, CopyComponent  # Replace your_module

initial_state = State(
    components=[
        CopyComponent(
            component_type="headline",
            element_type="headline_basic",
            char_limit=60,
            token_limit=15,
            audience="Sports Enthusiasts"
        ),
        # Add more CopyComponent instances as needed
    ],
    generated_content=[],
    validation_results=[],
    errors=[],
    attempt_count=0,
    status="initial",
    generation_history=[],
    output=[]
)

result = graph.invoke(initial_state)
print(result)
```

Replace `your_module` with the actual name of the Python file containing your code.

## Configuration

- **`.env` file:** Used to store sensitive information like API keys.
- **`template_kb.json`:** Contains the knowledge base used for generating prompts. You can customize this file to provide specific examples and constraints for different content components.
- **Logging:** The logging level and format can be configured in the `logging.basicConfig` call within the Python script.

## Docker

If you are running this agent within a Docker container, the generated output files will be located in the `/deps/__outer_studio/src/exports/` directory inside the container.

To copy the exports from the Docker container to your local machine, use the following command:

```bash
docker cp <container_id>:/deps/__outer_studio/src/exports/ <local_destination_path>
```

Replace `<container_id>` with the actual ID of your running Docker container and `<local_destination_path>` with the path on your local machine where you want to save the exports.

For example:

```bash
docker cp 5883f8f3010f:/deps/__outer_studio/src/exports/ /Users/mg/Desktop/GitHub/Playground/Agents/decathlon_agent/studio/exports/
```

## Contributing

Contributions to this project are welcome! Please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure they are well-documented.
4. Submit a pull request with a clear description of your changes.

## License

[Specify your project's license here]
```