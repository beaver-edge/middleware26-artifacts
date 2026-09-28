CONTEXT_PROMPT = """
You are an expert in Machine Learning on Edge Devices, highly skilled in the TinyML workflows, tools, techniques, and best practices. Your expertise covers both software and hardware, including microcontrollers and microprocessors. You will be asked questions related to various stages of the TinyML lifecycle, including data engineering, model design, model evaluation, model conversion and quantization, and deployment sketch development. The main task is to generate code to perform corresponding tasks (e.g., data cleaning, model quantization).

Your output will be executed **directly**, without additional checks or modifications. Therefore, it is critical that your code strictly adheres to the given format requirements and task instructions."""


SUGGESTION_RESPONSE_FORMAT = r"""
### RESPONSE FORMAT ###
- Output must be a SINGLE VALID JSON dictionary object containing key-value pairs where:
  - Keys: operation name (all lowercase, connected by _, no space allowed), 
  - Value: the operation description as short, clear English sentence. 
- Keep the answer short and concise, WITHOUT including any title, summary, question, or conclusion texts.
- The output MUST contain ONLY valid characters for ONE dictionary object, WITHOUT any characters outside of dictionary, WITHOUT line change characters
"""

# placeholders: dataset_intro, dataset_summary, purpose
# count_placeholders: 3
PROCESSING_SUGGESTIONS_PROMPT = r"""
### OBJECTIVE ###
Generate 8 practical data processing suggestions as key-value pairs for building a ML model to {purpose}.

### INSTRUCTIONS ###
- Provide exactly 8 suggestions in sequential order.
- Each suggestion should describe a useful **data engineering step** inspired by the application and dataset. 
- Do NOT include trivial steps such as "load dataset".
- Keep values concise, written as clear English sentences ending with a period.
- Keys must be in `lowercase_with_underscores`.


### USEFUL INFORMATION ###
- **DATASET INTRODUCTION**: 
{dataset_intro}
- **DATASET SUMMARY**: 
{dataset_summary}


### OUTPUT TEMPLATE ###
```json
{{"operation-name": "description for operation",..., "operation-name": "description for operation", ...}}
```
"""

# placeholders: executed_code, error_info, dataset_intro, dataset_summary, purpose
# count_placeholders: 5
PROCESSING_SUGGESTIONS_error_handling_PROMPT = r"""
### OBJECTIVE ###
Generate a new data processing suggestion table to avoid the error in the previous output reported under ### PREVIOUS RETURNED SUGGESTION TABLE ###. 

### INSTRUCTIONS ###
- Review the error details and regenerate the new suggestion table to prevent the same issue.  
- Provide exactly 8 practical data engineering suggestions in sequential order.  
- Each suggestion must describe a meaningful transformation inspired by the application and dataset.  
- Do NOT include trivial steps such as "load dataset".  
- Keys must be in `lowercase_with_underscores` and prefixed with step numbers.  
- Values must be short English sentences ending with a period.  
- Output must be a **SINGLE VALID JSON object** in one line, with no extra text or formatting.


### USEFUL INFORMATION ###
- **DATASET INTRODUCTION**: 
{dataset_intro}

- **DATASET SUMMARY**: 
{dataset_summary}

### PREVIOUS RETURNED SUGGESTION TABLE ###
```json
{executed_code}
```

### CAUSED ERROR ###
```
{error_info}
```

### OUTPUT TEMPLATE ###
```json
{{"operation-name": "description for operation",..., "operation-name": "description for operation", ...}}
```
"""



CODE_RESPONSE_FORMAT = r"""
### RESPONSE FORMAT ###
Your response must contain **ONLY TWO CODE BLOCKS** in the following order:      
- Output contains ONLY two code blocks:    
  1. **Python code block**      
    - Contains **ONLY IMPLEMENTATION CODE** (without any explanations, instructions, or examples).  
    - Must be directly executable as provided.  
    - Completed and self-contained in ```python\n...\n```
  2. **JSON code block**
    - Contains only the dataset paths updated by the Python code execution.
    - Paths must be **CONCRETE AND VALID**, reflecting actual paths created/modified by your code. 
    - Self-contained in ```json\n...\n```
- Provide clear, accurate, and directly executable code
- Never make assumptions - code should handle edge cases appropriately
- When requirements conflict with error information, **strictly follow the requirements**
- Answer WITHOUT assuming things, the code should be clear, accurate, and directly executable
- If error info conflicts with requirements, follow requirements strictly
-  <CRITICAL> Use concrete paths defined under **DATASET_PATH** instead of placeholders like "path/to/file" or "path/to/dataset.csv". PLACEHOLDER PATHS WILL ONLY CAUSE EXECUTION FAILURE </CRITICAL>
- Ensure JSON paths match exactly what your Python code creates/modifies, the returned paths should be really created when executing the code.
- If no dataset path is updated in your code, return the original dataset path provided."""


# placeholders: operation_description, current_operation_name, dataset_path, dataset_summary, completed_processing_steps
# count_placeholders: 5
OPERATION_CODE_PROMPT = r"""
### OBJECTIVE ###
Implement the specified data engineering operation in Python.

### INSTRUCTIONS ###
- Write complete, runnable Python code to implement the operation: {operation_description}.
- The current operation is: **{current_operation_name}**.
- Code must be self-contained with necessary imports.
- Read the dataset from the given path, not hardcoded values.
- Use only practical, minimal code (no placeholders or pseudocode).

### USEFUL INFORMATION ###
- **DATASET PATH**: 
Read dataset from this path: {dataset_path}
- **DATASET SUMMARY**: 
{dataset_summary}
- **PREVIOUS PROCESSING**: 
{completed_processing_steps}

### OUTPUT TEMPLATE ###
```python
<complete_code>
```
```json
{{
  <k-v pairs of updated dataset paths> 
}}
```
"""

# placeholders: current_operation, executed_code, error_info, dataset_path, dataset_summary, completed_processing_steps
# count_placeholders: 6
OPERATION_error_handling_PROMPT = r"""
### OBJECTIVE ###
Regenerate Python code to implement the current operation, fixing the error reported in ### CAUSED ERROR ###.

### INSTRUCTIONS ###
- Regenerate the code to avoid this error. 
- **CURRENT OPERATION**: {current_operation}
- Carefully review the caused error and adjust the code to resolve it.
- Perform a thorough code review before outputting.
- Read the dataset from the given path, not hardcoded values.
- Use only practical, minimal code (no placeholders or pseudocode).
- Ensure the Python code is self-contained and  directly runnable and imports all required libraries.

### EXECUTED CODE ###
```python
{executed_code}
```

### CAUSED ERROR ###
```
{error_info}
```

### USEFUL INFORMATION ###
- **DATASET PATH**: 
Read dataset from this path: {dataset_path}
- **DATASET SUMMARY**: 
{dataset_summary}
- **PREVIOUS PROCESSING**: 
{completed_processing_steps}

### OUTPUT TEMPLATE ###
```python
<complete_code>
```
```json
{{
  <k-v pairs of updated dataset paths> 
}}
```

"""
