CONTEXT_PROMPT = """
You are an expert in Machine Learning on Edge Devices, highly skilled in the TinyML workflows, tools, techniques, and best practices. Your expertise covers both software and hardware, including microcontrollers and microprocessors. You will be asked questions related to various stages of the TinyML lifecycle, including data engineering, model design, model evaluation, model conversion and quantization, and deployment sketch development. The main task is to generate code to perform corresponding tasks (e.g., data cleaning, model quantization).

Your output will be executed **directly**, without additional checks or modifications. Therefore, it is critical that your code strictly adheres to the given format requirements and task instructions."""


CODE_RESPONSE_FORMAT = r"""
### RESPONSE FORMAT ###
- Generate a Python script based *precisely* on the provided application description and requirements.
- Output should contain ONLY a code block where there is completed python code.
- The code should be clear, accurate, and directly executable, skip any code you are unsure about the detail.
- Variable and file naming should clearly reflect the purpose.
- If error info conflicts with my requirements, follow my requirements strictly while re-generating code.
- Your goal is to generate a complete, runnable Python script based on the user's application description and requirements. 
- Ensure the code is clean, well-commented, and uses standard Python libraries ('numpy', 'time', 'os', 'cv2' only if needed for image/video).
- The generated script should be enclosed in a single Python code block like ```python ... ```
- Provide the complete Python script enclosed in a single ```python ... ``` block, you cannot skip outputting any part of the code.
"""

# ! [ ] FIXME: **The first 10 lines of labelmap** now is hardcoded, should be a placeholder which to be filled by the first 10 lines read by the programe from the defined labelmap file path.
# Task prompt for generating the Python sketch
PYTHON_SKETCH_PROMPT = r"""
### OBJECTIVE ###
Generate a complete Python script for the application named '{application_name}'. The deployment target device is '{target_device}'.


### CONFIGURATION PARAMETERS ###
All the required paths and parameters are provided below, use them in generated code:
- Input Method Description: '{input_description}'
- Output Method Description: '{output_description}'
- model_path = '{model_path}'
- label_path = '{label_path}'  
- input_path = '{input_path}'  
- output_path = '{output_path}' 
- confidence_threshold = '{confidence_threshold}'

### USEFUL INFORMATION ###
**The first 10 lines of labelmap**:
```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
```

### INSTRUCTIONS ###
- Adhere strictly to the Application Description and Core Requirements variablesw provided above to generate the complete Python script.
- Use the paths and parameters provided under ### CONFIGURATION PARAMETERS ###, no placeholders allowed in your code. All the needed paths and parameters are already included.
- Follow the phase structure outlined in the ### PROGRAMMING GUIDELINES ### reference logic provided below.
- Implement the specific Python code for Phases 2, 4.2, and 4.3 based *solely* on the Application Description and Input/Output Method Descriptions provided above.
- Import standard libraries only as needed for the described application logic.
- Import ONLY the libraries needed for the specific task defined by the application description and I/O methods. Do not assume libraries like 'cv2' unless image/video processing is explicitly mentioned.
- The script must be optimized for {target_device} EdgeTPU execution with proper EdgeTPU delegate usage.
- Include error handling for EdgeTPU delegate loading failures and provide informative error messages.
- Check to insure you have NO placeholders like "path_to..." in your code, all the real values of paths and parameters are already included under ### CONFIGURATION PARAMETERS ###.

### PROGRAMMING GUIDELINES ###
{core_logic_reference_formatted}

"""

# Task prompt for handling errors in generated code
PYTHON_SKETCH_error_handling_PROMPT = r"""
### OBJECTIVE ###
- The previous Python script generated has error as logged under ### CAUSED ERROR BEFORE ###.
- The "CAUSED ERROR BEFORE" lists the error message and your previous faulty code caused the error is under ### EXECUTED CODE BEFORE  ###.
- Following the phase structure outlined in the ### PROGRAMMING GUIDELINES ### reference logic provided below, REWRITE the complete Python script for the application '{application_name}' to avoid the error.
- *MAKE SURE YOU USED THE PROVIDED PATHS AND PARAMETERS*


### CONFIGURATION PARAMETERS ###
All the required paths and parameters are provided below, use them in generated code:
- Input Method Description: '{input_description}'
- Output Method Description: '{output_description}'
- model_path = '{model_path}'
- label_path = '{label_path}'  
- input_path = '{input_path}'  
- output_path = '{output_path}' 
- confidence_threshold = '{confidence_threshold}'

### USEFUL INFORMATION ###
**The first 10 lines of labelmap**:
```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
```

### INSTRUCTIONS ###
-  The target deployment device fot the code is {target_device} with EdgeTPU.
- Adhere strictly to the Application Description and Core Requirements variables provided above.
- Follow the phase structure outlined in the ### PROGRAMMING GUIDELINES ### reference logic provided below.
- Always implement EdgeTPU acceleration using the load_delegate mechanism with proper fallback handling.
- Implement the specific Python code for Phases 2, 4.2, and 4.3 based *solely* on the Application Description and Input/Output Method Descriptions provided above.
- **Load and use the Label Map Path IF it is provided AND relevant** to interpreting the model's output, as suggested by the Application Description.
- Import standard libraries only as needed for the described application logic.
- Import ONLY the libraries needed for the specific task defined by the application description and I/O methods. Do not assume libraries like 'cv2' unless image/video processing is explicitly mentioned.
- Include error handling for EdgeTPU delegate loading failures and provide informative error messages.
- Check to insure you have NO placeholders like "path_to..." in your code, all the real values of paths and parameters are already included under ### CONFIGURATION PARAMETERS ###.


### PROGRAMMING GUIDELINES ###
{core_logic_reference_formatted}

### CAUSED ERROR BEFORE ###
{error_message}

### EXECUTED CODE BEFORE ###
```python
{faulty_code}
```

### OUTPUT TEMPLATE ###
```python
<complete_code>
```
<REPORT THE FIX OF THE LAST ERROR>
"""

# 1.7. **Model Compatibility Check**: For object detection models, check output layer names to determine TF1 vs TF2 model format and set appropriate output indices: `if 'StatefulPartitionedCall' in output_details[0]['name']: boxes_idx, classes_idx, scores_idx = 1, 3, 0` else `boxes_idx, classes_idx, scores_idx = 0, 1, 2`.
# 


CORE_LOGIC_REFERENCE =r"""{
"Programming Guidelines for TFLite Inference with EdgeTPU": {
  "Phase 1: Setup": {
 
      "1.1": {"Imports": "Import interpreter and delegate literally by `from tflite_runtime.interpreter import Interpreter, load_delegate`. If import fails, fallback to `from tensorflow.lite import Interpreter` and `from tensorflow.lite.experimental import load_delegate`."},
      "1.2": {"Paths/Parameters": "Define necessary variables using the model path, input path (if provided and relevant), label path (if provided and relevant), output paths, or other parameters provided in the main prompt."},
      "1.3": {"Load Labels (Conditional)": "If a label path variable was defined and is needed for the application (check main prompt's description), implement code to read the label file into a Python list."},
      "1.4": {"Load Interpreter with EdgeTPU": "Instantiate interpreter with EdgeTPU acceleration:
            ```
            try:
                interpreter = Interpreter(model_path=..., 
                                        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            except:
                interpreter = Interpreter(model_path=..., 
                                        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
            ```
            using the provided model path variable, and call `interpreter.allocate_tensors()`."},
      "1.5": {"Get Model Details": "Retrieve `input_details = interpreter.get_input_details()` and `output_details = interpreter.get_output_details()`. Store necessary indices and properties (shape, dtype) for input/output tensors."}
   
  },
  "Phase 2: Input Acquisition & Preprocessing Loop": {
    
      "2.1": {"Acquire Input Data": "Implement code to get raw data according to the input description provided in the main prompt. Use the input path variable if it was provided and is relevant to the described input method. This phase is highly application-dependent."},
      "2.2": {"Preprocess Data": "Implement code to transform the raw data into the `numpy` array(s) based on the extracted information of shape and `dtype` from the retrieved `input_details`. Store result in an `input_data` variable."},
      "2.3": {"Quantization Handling": "Check if model uses floating point inputs: `floating_model = (input_details[0]['dtype'] == np.float32)`. If floating model, normalize: `input_data = (np.float32(input_data) - 127.5) / 127.5`."},
      "2.4": {"Loop Control": "If the input description implies processing continuous data (stream, camera, multiple files), implement the appropriate loop structure and exit conditions. If processing single input, plan to run Phase 3 once."}
   
  },
  "Phase 3: Inference": {
    
      "3.1": {"Set Input Tensor(s)": "For each required input tensor: `interpreter.set_tensor(input_details[idx]['index'], input_data_variable[idx])`. Use the preprocessed `input_data` variable."},
      "3.2": {"Run Inference": "`interpreter.invoke()`."}
   
  },
  "Phase 4: Output Interpretation & Handling Loop": {
    
      "4.1": {"Get Output Tensor(s)": "Retrieve results using `output_variable[idx] = interpreter.get_tensor(output_details[idx]['index'])`. Store in an `output_data` variable."},
      "4.2": {"Interpret Results": "Implement code to process the raw `output_data` variable according to the specific task to generate meaningful results. If labels were loaded in Phase 1.3 and are relevant, use the label list here to map indices to names."},
      "4.3": {"Post-processing": "For detection models, apply confidence thresholding, coordinate scaling, and bounding box clipping to ensure results are within valid ranges."},
      "4.4": {"Handle Output": "Implement code to deliver the interpreted results according to the output description provided in the main prompt (e.g., print, write to file, send data, control actuator)."},
      "4.5": {"Loop Continuation": "If inside a loop from Phase 2.4, implement logic to continue or break based on exit conditions."}
   
  },
  "Phase 5: Cleanup": {
    
      "5.1": {"Resources Release": "Implement code to release any resources acquired in Phase 2 (e.g., close files, release camera)."}
  }
}}
"""
 
