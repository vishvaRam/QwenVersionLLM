# Image to Text Extraction using Qwen2-VL Model

This project provides a Python script that utilizes the Qwen2-VL model to extract handwritten details from payment receipt vouchers. The model processes images of receipts and retrieves specific information such as names, receipt numbers, dates, rupees, amounts in words, and reasons for payment.

## Memory Management

The `ImageToTextModel` class is designed to optimize memory usage by ensuring that only one instance of the model and its processor is loaded into memory. This is achieved through the use of class variables, which cache the model and processor after their initial loading. 

By doing so, the class prevents redundant loading of the model, saving both time and computational resources. This approach is particularly beneficial in scenarios where the model is accessed multiple times, as it significantly reduces the overhead associated with loading large models repeatedly.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [How It Works](#how-it-works)
- [Contributing](#contributing)


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name


2. Create a virtual environment (optional but recommended):
   
  ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
3. Install the required dependencies:
    Need to install transformers from git
    And also other packages.
  ```bash
  # Need to install transformers from git
  pip install git+https://github.com/huggingface/transformers
  
  ```


## Usage

To use the script for extracting information from a payment receipt:

1. Import the `ImageToTextModel` class from the script.

2. Create an instance of the class and call the `img_to_text` method, providing the path to the image and the prompt.

   ```python
   from your_script_name import ImageToTextModel

   model = ImageToTextModel()
   img_path = "path/to/your/image.jpg"
   prompt = """
       Extract the following handwritten details from the payment receipt voucher from given image:
           - Name: Identify the handwritten name on the voucher.
           - Receipt Number: Extract the handwritten receipt number.
           - Date: Identify the handwritten date as "dd/mm/yy" on the voucher.
           - Rupees: Extract the handwritten rupees (Rs) written in numbers.
           - Amount in Words: Extract the handwritten amount written in words.
           - Reason for Payment: Identify the handwritten reason for the payment.
           Ensure accuracy in capturing all handwritten details.
           Do not generate any text or sentence above or below.
   """

   extracted_details = model.img_to_text(img_path, prompt)
   print(extracted_details)


## Dependencies

This project requires the following Python packages:

- `torch`
- `transformers`
- `qwen_vl_utils`
- `timer`

Make sure to install these dependencies via the `requirements.txt` file.

## How It Works

1. **Model Initialization**: The `ImageToTextModel` class loads the Qwen2-VL model and its processor. It caches the model and processor to optimize performance.

2. **Input Preparation**: The `_prepare_inputs` method prepares the input data, including the image and the prompt.

3. **Text Generation**: The `_generate_text` method generates the output text based on the prepared inputs.

4. **Entity Extraction**: The `extract_entities` method uses regular expressions to extract specific details from the generated output.

5. **Image to Text Conversion**: The `img_to_text` method combines all the above steps to process the image and return the extracted details as a list.

## Contributing

Contributions are welcome! Please create a pull request for any enhancements or bug fixes.


