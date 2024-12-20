## Decathlon Studio - Development Plan

**Goal:** Develop a working Proof of Concept (POC) for automated content generation and validation.

**Key Principle:** Focus on core functionality for the POC, deferring non-essential enhancements.

### Phase 1: Core Structure and State Management

-   **Step 1: Knowledge Base Integration (Basic):**
    -   [x] Create `template_kb.json` (simplified version)
    -   [x] Load `template_kb.json` in `decathlon_studio.py`
    -   [x] Modify `generate_content` to accept `knowledge_base`
    -   [x] Update prompt construction to use basic data from `knowledge_base` (e.g., `char_limit`, `token_limit`)
-   **Step 2: Refactor State:**
    -   [x] Convert `CopyComponent` to a `dataclass`
    -   [x] Update `State` class (add `generation_history`, `errors`, `status`)
    -   [x] Remove `InputSchema` and handle components directly
    -   [x] Adjust `example_input`
-   **Step 3: Enhance Prompt to Provide Feedback:**
    -   [x] Basic implementation to inform about validation failure.
    -   [ ] **TODO (For POC):** Refine prompt in `generate_content` to provide specific feedback when validation fails (e.g., "Content exceeded character limit by X characters.", "Content exceeded token limit by Y tokens.").

### Phase 2: Enhanced Validation and Workflow

-   **Step 4: Implement Detailed Validation:**
    -   [x] Refactor `validate_content` to perform checks for character limits and token limits.
    -   [ ] **TODO (For POC):** Generate detailed feedback using `generate_feedback` to provide user-friendly error messages about validation failures.
    -   [ ] **TODO (For POC):** **Address prefix handling:** Ensure validation considers the actual content length after removing potential prefixes like "Text:" or "Content:". This might involve a cleaning step within `validate_content` or adjusting the length calculation.
    -   [ ] **TODO (Post-POC):** Add validation for duplicate content (compare against `generation_history`).
-   **Step 5: Refine `should_continue`:**
    -   [x] Update `should_continue` to use the `status` and `validation_results` fields.
    -   [x] Ensure it correctly determines whether to regenerate content or end the process based on validation and attempt count.

### Phase 3: Export and Advanced Features (Consider for POC)

-   **Step 6: Implement Export Functionality:**
    -   [ ] **TODO (Consider for POC):** Create an `export_results` function to save generated content and metadata to a JSON file. Evaluate if essential for demonstrating core POC functionality.
    -   [ ] **TODO (If included in POC):** Add a new node called `export` to the graph.
    -   [ ] **TODO (If included in POC):** Connect the `validate` node to `export` when content is successfully validated.
-   **Step 7: Clean Content Function (Conditional for POC):**
    -   [ ] **TODO (Conditional for POC):** Implement a `clean_content` function to remove prefixes like "Text:" or "Content:" from the generated output **if this is a consistent issue affecting validation and readability.**  Consider integrating this within `generate_content` or `validate_content`.
-   **Step 8: Token Counting and Limit Validation (Post-POC):**
    -   [ ] **TODO (Post-POC):** Integrate the `count_tokens` and `estimate_tokens_by_char_count` functions for accurate token calculation.
    -   [ ] **TODO (Post-POC):** Ensure `validate_content` utilizes these functions for precise token limit validation.
-   **Step 9: Externalize Knowledge Base (Post-POC):**
    -   [ ] **TODO (Post-POC):** Move the `template_kb.json` to the `knowledge_base` folder.
    -   [ ] **TODO (Post-POC):** Create a `template_kb.py` file to handle loading and querying the knowledge base.
    -   [ ] **TODO (Post-POC):** Modify `decathlon_studio.py` to import and use the external knowledge base.
-   **Step 10: Dynamic Knowledge Base Updates (Post-POC):**
    -   [ ] **TODO (Post-POC):** Implement a mechanism to update the knowledge base dynamically without restarting the agent.

**Next Steps:**

We will now work through the items marked as `TODO (For POC)` in the plan, starting with enhancing the prompt feedback and then moving on to detailed validation, ensuring we address the prefix handling during the validation process.