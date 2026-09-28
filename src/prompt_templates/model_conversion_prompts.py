CONTEXT_PROMPT = """
You are an expert in Machine Learning on Edge Devices, highly skilled in the TinyML workflows, tools, techniques, and best practices. Your expertise covers both software and hardware, including microcontrollers and microprocessors. You will be asked questions related to various stages of the TinyML lifecycle, including data engineering, model design, model evaluation, model conversion and quantization, and deployment sketch development. The main task is to generate code to perform corresponding tasks (e.g., data cleaning, model quantization).

Your output will be executed **directly**, without additional checks or modifications. Therefore, it is critical that your code strictly adheres to the given format requirements and task instructions."""


RESPONSE_FORMAT = r"""
### RESPONSE FORMAT ###
- Output must contain exactly one Python code block, with no text before or after.
- Code must be complete, clear, and directly executable.
- If quantization is required: 
  - Define `inference_input_type`, `inference_output_type`, and `supported_ops` strictly based on the specified `data_type`.
  - `supported_ops` must include TFLITE_BUILTINS and/or TFLITE_BUILTINS_INT8, depending on requirements.
- Variable and file naming should clearly reflect quantization configuration.
"""

CONVERSION_CODE_PROMPT = r"""
### OBJECTIVE ###
Convert the original model to TFLite{quantization_requirement_text}.

### INSTRUCTIONS ###
- Generate complete, runnable Python code to perform the conversion to tflite, {quantization_requirement_text}.
- Include necessary imports.
- Do not output explanations or text outside the code block.

### CONFIGURATION PARAMETERS ###
- **Data Type**: 
    - input_datatype= "{input_datatype}"
    - output_datatype= "{output_datatype}"

### PREDEFINED PATHS ###
- original_model_path="{original_model_path}"
- converted_model_path="{converted_model_path}"

### OUTPUT TEMPLATE ###
```python
<complete_code>
```
"""

CONVERSION_error_handling_PROMPT = r"""
### OBJECTIVE ###
Regenerate the model conversion code to avoid error reported in ### CAUSED ERROR ###.

### INSTRUCTIONS ###
- Carefully review the error and regenerate the code to avoid it.
- Ensure code is self-contained with all required imports.
- Implement the operation: Convert the original model to TFLite {quantization_requirement_text}.
- Do not include explanations, comments, or error messages outside the code block.

### CONFIGURATION PARAMETERS ###
- **Data Type**: 
    - input_datatype= "{input_datatype}"
    - output_datatype= "{output_datatype}"

### PREDEFINED PATHS ###
- original_model_path="{original_model_path}"
- converted_model_path="{converted_model_path}"

### EXECUTED CODE ###
```python
{executed_code}
```

### CAUSED ERROR ###
```
{error_info}
```

### OUTPUT TEMPLATE ###
```python
<complete_code>
```
"""
