CONTEXT_PROMPT = """
You are an expert in Machine Learning on Edge Devices, highly skilled in the TinyML workflows, tools, techniques, and best practices. Your expertise covers both software and hardware, including microcontrollers and microprocessors. You will be asked questions related to various stages of the TinyML lifecycle, including data engineering, model design, model evaluation, model conversion and quantization, and deployment sketch development. The main task is to generate code to perform corresponding tasks (e.g., data cleaning, model quantization).

Your output will be executed **directly**, without additional checks or modifications. Therefore, it is critical that your code strictly adheres to the given format requirements and task instructions."""



SPECIFICATION_RESPONSE_FORMAT = r"""
### RESPONSE FORMAT ###
- Output MUST contain exactly one code block, and nothing else.
- The code block must include ONLY the updated `"application_specifications":{{...}}` object.
- <CRITICAL> Retain the placeholder `"programming_guidelines": {programming_guidelines}` exactly as provided.</CRITICAL>
- The output should be self-contained with no information loss.
"""

APPLICATION_SPEC_FILL_PROMPT = r"""
### OBJECTIVE ###
Fill in the requested fields of `application_specifications`.

### INSTRUCTIONS ###
- Read `"board_fullname"` from `"hardware"`.
- Based on board, application description, and sensors, fill in the placeholder fields:
  - `{{decide_when_generating_code_based_on_given_board_and_application_description}}` → values determined by board/application.
  - `{{decide_when_generating_code_based_on_given_data_sample_and_application_description}}`  → values determined by dataset/application.
- Keep `"programming_guidelines"` unchanged.
- Ensure libraries are compatible with `{board_fullname}`.
- Use `"TensorFlowLite.h"` instead of `"Arduino_TensorFlowLite.h"`.

### DATASET INFORMATION ###
**DATASET SUMMARY**: 
{dataset_summary}

### TARGET TO BE FILLED ###
```json
{application_spec_template}
```
"""

# placeholders: dataset_summary, board_fullname, executed_code, error_info, application_spec_template
# count_placeholders: 5
APPLICATION_SPEC_error_handling_PROMPT = r"""
### OBJECTIVE ###
Regenerate the `application_specifications` to avoid errors in the previously returned version.

### INSTRUCTIONS ###
-. Review the caused error and format rules.
-. If any required fields were missing, add them back.
- Read `"board_fullname"` from `"hardware"`.
- Based on board, application description, and sensors, fill in the placeholder fields:
  - `{{decide_when_generating_code_based_on_given_board_and_application_description}}` → values determined by board/application.
  - `{{decide_when_generating_code_based_on_given_data_sample_and_application_description}}`  → values determined by dataset/application.
- Keep `"programming_guidelines"` unchanged.
- Ensure libraries are compatible with `{board_fullname}`.
- Use `"TensorFlowLite.h"` instead of `"Arduino_TensorFlowLite.h"`.

### DATASET INFORMATION ###
**DATASET SUMMARY**: 
{dataset_summary}

### PREVIOUS RETURNED APPLICATION SPECIFICATIONS ###
```json
{executed_code}
```

### CAUSED ERROR ###
```
{error_info}
```

### TARGET TO BE FILLED ###
```json
{application_spec_template}
```
"""


CODE_RESPONSE_FORMAT = r"""
### RESPONSE FORMAT ###
- Output MUST contain EXACTLY ONE code block with valid sketch code in C++ (.ino format).
- <CRITICAL>The sketch MUST include the path to the model file: `#include "model.h"`.</CRITICAL>
- If any detail is uncertain, skip it rather than guessing.
- Only code is needed in the output
- Adding anything outside the single code block will break the program.
"""

# placeholders: dataset_summary, application_spec_template
# count_placeholders: 2
ARDUINO_SKETCH_PROMPT = r"""
### OBJECTIVE ###
Generate an Arduino .ino sketch in C++ for the described application.

### INSTRUCTIONS ###
- Follow the `programming_guidelines` property from the `application_specification`, which is a programming guidelines for arduino sketches.
- Include `#include "model.h"`.
- Ensure the sketch is clear, correct, and executable.

### DATASET INFORMATION ###
- **DATASET SUMMARY**: {dataset_summary} 

### APPLICATION SPECIFICATIONS ###
```json
{application_spec_template}
```

### OUTPUT TEMPLATE ###
```cpp
<complete_sketch_code>
```
"""

# placeholders: dataset_summary, executed_code, error_info, application_spec_template
# count_placeholders: 4
ARDUINO_SKETCH_error_handling_PROMPT = r"""
### OBJECTIVE ###
Regenerate the .ino sketch code for the application to avoid the error shown below.

### INSTRUCTIONS ###
- Follow the `programming_guidelines` property from the `application_specification`, which is a programming guidelines for arduino sketches.
- Modify ONLY what is necessary to resolve the error.
- Ensure the sketch compiles for `{board_fullname}`.
- Always include: `#include "model.h"`.

### EXECUTED CODE BEFORE ###
```cpp
{executed_code}
```

### CAUSED ERROR BEFORE ###
```
{error_info}
```

### DATASET INFORMATION ###
- **DATASET SUMMARY**: {dataset_summary}

### APPLICATION SPECIFICATIONS ###
```json
{application_spec_template}
```

### OUTPUT TEMPLATE ###
```cpp
<complete_sketch_code>
```
"""

APPLICATION_SPEC_TEMPLATE = r"""{{
"application_specifications": {{
    "application": {{
        "name": "{application_name_hinting_its_purpose}",
        "description": "{application_description}"}},
    "hardware": {{
        "board": "{board_fullname}",
        "sensors": {{"{decide_when_generating_code_based_on_given_board_and_application_description}": "{decide_when_generating_code_based_on_given_board_and_application_description}"}}}},
    "software": {{
        "libraries": {{
            "main_library": {{
                "name": "TensorFlowLite",
                "header": "TensorFlowLite.h"}},
            "other_libraries": [
                {{
                    "name": "{decide_when_generating_code_based_on_given_board_and_application_description}",
                    "header": "{decide_when_generating_code_based_on_given_board_and_application_description}.h"
                }},
                {{
                    "name": "{decide_when_generating_code_based_on_given_board_and_application_description}",
                    "header": "{decide_when_generating_code_based_on_given_board_and_application_description}.h"
                }},
                {{{decide_when_generating_code_based_on_given_board_and_application_description}}},
                {decide_when_generating_code_based_on_given_board_and_application_description}]
        }},
        "model": {{
            "path": "./model.h",
            "input_tensor": {{
                "dimensions": "{decide_when_generating_code_based_on_given_data_sample_and_application_description}",
                "data_type": "{input_datatype}"
            }},
            "output_tensor": {{
                "dimensions": "{decide_when_generating_code_based_on_given_data_sample_and_application_description}",
                "data_type": "{output_datatype}"
            }},
            "tensor_arena_size": "{decide_when_generating_code_based_on_given_data_sample_and_application_description}"
        }}
    }},
    "deployment": {{
        "device": "{board_fullname}",
        "communication_interface": "Serial",
        "baud_rate": 9600
    }},
    "classification": {{
        "classes": {classification_classes}
    }},
    "programming_guidelines": {programming_guidelines},}}}}"""

# Programming Guidelines template outlining necessary steps for sketch code
PROGRAMMING_GUIDELINES_TEMPLATE = r"""{
"Programming Guidelines for TFLite Micro Inference on Microcontrollers": {
"Phase 1": {
  "Initialization": {
    "1.1": {"Include Necessary Libraries": "Use the exact vendored headers: `TensorFlowLite.h`, `tensorflow/lite/micro/all_ops_resolver.h`, `tensorflow/lite/micro/micro_interpreter.h`, `tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h`, `tensorflow/lite/schema/schema_generated.h`, `Arduino_APDS9960.h`, and `model.h`. Do not include the obsolete `tensorflow/lite/micro/micro_error_reporter.h` path."},

    "1.2": {"Declare Variables": "The supplied `model.h` declares the byte array as `model`; load it with `tflite::GetModel(model)` (never `g_model`). Use `tflite::AllOpsResolver`, `tflite::MicroInterpreter`, and `TfLiteTensor`. This vendored `MicroInterpreter` constructor takes `(model, resolver, tensor_arena, tensor_arena_size)` without an error-reporter argument. If logging is needed, use `tflite::ErrorReporter* error_reporter = tflite::GetMicroErrorReporter()`."},
    "1.3": {"Define Tensor Arena": "Define and allocate a tensor arena buffer with a carefully chosen size. Ensure sufficient memory is available to prevent crashes while avoiding unnecessary waste."},
    "1.4": {"Load the Model": "Load the model from the given path using the provided TFLite Micro function and check for validity."},
    "1.5": {"Resolve Operators": "Register the necessary operators using an `OpResolver`. Use specific operators if model architecture is known. Otherwise, use `AllOpsResolver` as a fallback."},
    "1.6": {"Instantiate the Interpreter": "Initialize the interpreter with the model, operator resolver, tensor arena, and error reporter."},
    "1.7": {"Allocate Memory": "Allocate memory for tensors using the tensor arena. Perform error checking to confirm success."},
    "1.8": {"Define Model Inputs": "Retrieve model input and output tensors with `interpreter->input(0)` and `interpreter->output(0)`. Use the vendored union members such as `tensor->data.f`, `tensor->data.uint8`, or `tensor->data.int8`; there is no `data.u8` member."},
    "1.9": {"Set Up Other Relevant Parts": "For the Nano 33 BLE Sense color sensor, use the library's global `APDS` object. Initialize only with `APDS.begin()`, wait for `APDS.colorAvailable()`, and read using `APDS.readColor(int& r, int& g, int& b)`. Do not call private `enableColor`, nonexistent `enableColorSensor`, or one-argument color-read methods."}
  }
},
"Phase 2": {
  "Preprocessing": {
    "2.1": {"Sensor Setup": "Initialize and configure the sensor(s) or input data source(s). Ensure data acquisition works correctly."},
    "2.2": {"Optional Feature Extraction": "If needed, apply transformations or feature extraction on raw sensor data before feeding it to the model."}
  }
},
"Phase 3": {
  "Inference": {
    "3.1": {"Data Copy": "Copy the processed data (from sensor buffer or extracted features) into the model's input tensor buffer."},
    "3.2": {"Invoke Interpreter": "Run the inference using `interpreter.Invoke()`. Check return codes if applicable."}
  }
},
"Phase 4": {
  "Postprocessing": {
    "4.1": {"Process Output": "Interpret the model's output tensor(s) according to application requirements. Apply thresholds, mappings, or scaling if needed."},
    "4.2": {"Execute Application Behavior": "Trigger application-specific actions or behaviors based on the inference result (e.g., display output, send signal)."}
  }},}}
"""
