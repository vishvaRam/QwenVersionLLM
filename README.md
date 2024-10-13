# Image to Text Extraction using Qwen2-VL Model

This project provides a Python script that utilizes the Qwen2-VL model to extract handwritten details from payment receipt vouchers. The model processes images of receipts and retrieves specific information such as names, receipt numbers, dates, rupees, amounts in words, and reasons for payment.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

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
