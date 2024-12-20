# Decathlon CRM Copywriter POC - Revised Plan

## Phase 1: Core Structure and State Management (Completed)

- **Status:** Completed
- **Goal:** Establish the basic workflow, state management, and knowledge base integration.
- **Steps:**
  1. Create basic graph with `generate` and `validate` nodes. (Completed)
  2. Refactor `State` to use `TypedDict`. (Completed)
  3. Update `generate_content` to use knowledge base. (Completed)
  4. Integrate basic validation logic. (Completed)
  5. Implement `should_continue` logic. (Completed)
  6. Resolve JSON input issues. (Completed)
  7. Fix `attempt_count` incrementing and loop termination. (Completed)

## Phase 2: Enhanced Validation and Workflow Refinement (In Progress)

- **Status:** In Progress
- **Goal:** Improve validation, provide detailed feedback, and refine the workflow.
- **Steps:**

  1. **Refine Prompt Feedback:**
     - **Status:** TODO
     - **Description:** Enhance the prompt in `generate_content` to include specific feedback based on validation errors.
     - **Tasks:**
       - Modify `generate_content` to check `state["status"]`.
       - If `state["status"]` is "validation_failed", add a feedback section to the prompt.
       - Use information from `state["validation_results"]` to provide specific feedback (e.g., "Content exceeded character limit by X characters.").
       - Use information from `state["errors"]` to provide specific feedback.
       - Experiment with different phrasing and emphasis to guide the model.
     - **Expected Outcome:** The prompt dynamically adapts based on previous validation failures, providing specific guidance to the language model.

  2. **Implement Detailed Validation Feedback:**
     - **Status:** TODO
     - **Description:** Create a mechanism to generate user-friendly feedback messages based on validation results.
     - **Tasks:**
       - Create a new function `generate_feedback(validation_results: Dict[str, Any]) -> List[str]` that takes the `validation_results` dictionary as input.
       - Inside `generate_feedback`, analyze the `validation_results` and create descriptive error messages (e.g., "Content is too long.", "Content is empty.").
       - Return a list of error messages.
       - Modify `validate_content` to call `generate_feedback` and store the returned messages in the `errors` field of the state.
     - **Expected Outcome:** The `errors` field in the state contains a list of human-readable error messages explaining why the content failed validation.

  3. **Address Prefix Handling:**
      - **Status:** TODO
      - **Description:** Implement a mechanism to remove any unwanted prefixes (e.g., "Text:", "Content:") that the language model might add to the generated content.
      - **Tasks:**
          - Update `validate_content` or create a separate function `clean_content` to remove unwanted prefixes.
          - Update the prompt to explicitly instruct the model not to include prefixes.
          - Consider using regular expressions for flexible prefix removal.
      - **Expected Outcome:** The generated content is free of unwanted prefixes, ensuring accurate validation.

  4. **Test and Refine:**
     - **Status:** TODO
     - **Description:** Thoroughly test the workflow with various inputs and knowledge base entries. Refine the prompt, validation logic, and feedback mechanism as needed.
     - **Tasks:**
       - Run the workflow in LangGraph Studio with different `CopyComponent` inputs.
       - Analyze the generated content, validation results, and feedback messages.
       - Adjust the prompt, knowledge base, and code based on the observations.
     - **Expected Outcome:** The workflow consistently generates content that meets the specified criteria and provides helpful feedback when validation fails.

## Phase 3: Export and Advanced Features (Pending)

- **Status:** Pending
- **Goal:** Implement export functionality and explore advanced features.
- **Steps:**

  1. **Implement Export Functionality:**
     - **Status:** TODO
     - **Description:** Create a mechanism to export the generated content, validation results, and other relevant metadata to a JSON file.
     - **Tasks:**
       - Create a new function `export_results(state: State, filepath: str)` that takes the state and a filepath as input.
       - Inside `export_results`, extract the relevant data from the state (e.g., `generated_content`, `validation_results`, `errors`, `attempt_count`, `generation_history`).
       - Format the data as a JSON string.
       - Write the JSON string to the specified filepath.
       - Add an `export` node to the LangGraph workflow.
       - Connect the `validate` node to the `export` node, so that export is triggered when `status` is "completed" or "max_attempts_reached".
     - **Expected Outcome:** The workflow generates a JSON file containing the results of each run.

  2. **Implement Content Cleaning (Optional):**
     - **Status:** TODO
     - **Description:** Create a function to clean the generated content, removing any remaining unwanted artifacts or formatting issues.
     - **Tasks:**
       - Research and implement appropriate text cleaning techniques.
       - Integrate the cleaning function into the workflow, possibly as a separate node or within `generate_content`.
     - **Expected Outcome:** The generated content is further refined and polished.

  3. **Implement Token Counting and Limit Validation (Optional):**
     - **Status:** TODO
     - **Description:** Add validation logic to check if the generated content is within the specified token limit.
     - **Tasks:**
       - Research and implement a suitable token counting method.
       - Integrate token counting into `validate_content`.
       - Add a "within_token_limit" field to the `validation_results`.
     - **Expected Outcome:** The workflow validates both character and token limits.

  4. **Explore External Knowledge Base Management (Post-POC):**
     - **Status:** TODO
     - **Description:** Investigate options for managing the knowledge base outside of the code (e.g., in a separate database or file).
     - **Tasks:**
       - Research different knowledge base storage and retrieval mechanisms.
       - Design a suitable interface for interacting with the external knowledge base.
     - **Expected Outcome:** The knowledge base can be updated and managed independently of the workflow code.

  5. **Implement Dynamic Knowledge Base Updates (Post-POC):**
      - **Status:** TODO
      - **Description:** Implement a mechanism to dynamically update the knowledge base during workflow execution.
      - **Tasks:**
          - Design a strategy for triggering knowledge base updates (e.g., based on validation failures or user feedback).
          - Implement the update mechanism, potentially using an external API or database.
      - **Expected Outcome:** The workflow can adapt and improve its performance over time by learning from its mistakes and incorporating new information.