import os
import glob
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from timer import Timer
import warnings
import re
warnings.filterwarnings("ignore")


class ImageToTextModel:
    # Class variables to hold the model and processor, ensuring they are loaded only once
    _model = None
    _processor = None

    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        # Initialize the model and processor if not already loaded
        self._load_model_and_processor(model_name)

    @classmethod
    def _load_model_and_processor(cls, model_name):
        """Load the model and processor, ensuring they are cached for reuse."""
        if cls._model is None or cls._processor is None:
            print("Loading model and processor...")
            cls._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=r"D:\PythonModel"
            ).to('cuda')
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            cls._processor = AutoProcessor.from_pretrained(model_name,
                                                           min_pixels=min_pixels,
                                                           max_pixels=max_pixels,
                                                           cache_dir=r"D:\PythonModel")
            torch.cuda.empty_cache()  # Clear CUDA memory after loading the model
        else:
            print("\nModel and processor already loaded, using cached versions.")

    def _prepare_inputs(self, img_path, prompt):
        """Prepare the input data for the model: text and image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path, "max_pixels": 1280*28*28 },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare text and image inputs using the processor
        text = ImageToTextModel._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        inputs = ImageToTextModel._processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        )
        return inputs.to("cuda")  # Move inputs to CUDA

    def _generate_text(self, inputs):
        """Generate the output text based on the processed inputs."""
        generated_ids = ImageToTextModel._model.generate(**inputs, max_new_tokens=256,temperature=0.8)

        # Trim the generated output to get the relevant text
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the output text
        output_text = ImageToTextModel._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def extract_entities(self, text):
        # Use regular expressions to extract the values
        name = re.search(r'Name: (.+)', text).group(1)
        receipt_number = re.search(r'Receipt Number: (.+)', text).group(1)
        date = re.search(r'Date: (.+)', text).group(1)
        rupees = re.search(r'Rupees: (.+)', text).group(1)
        amount_in_words = re.search(r'Amount in Words: (.+)', text).group(1)
        reason_for_payment = re.search(r'Reason for Payment: (.+)', text).group(1)

        receipt_number_trimmed_text = receipt_number.replace(" ", "")
        # Create a list of entities
        entities = [name.upper(), date, rupees, receipt_number_trimmed_text.upper(), reason_for_payment, amount_in_words]

        return entities

    def img_to_text(self, img_path, prompt):
        timer = Timer()

        """Main method to process an image and prompt, and return the extracted text."""
        timer.start()
        inputs = self._prepare_inputs(img_path, prompt)  # Prepare inputs
        output_text = self._generate_text(inputs)  # Generate text from inputs
        timer.stop()

        list_output = self.extract_entities(output_text)
        return list_output

prompt ="""
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

# Create an instance of the ImageToTextModel class
# model_instance = ImageToTextModel()

# Call the method to convert image to text
# output = model_instance.img_to_text("D:\Vishva\VersionLLM\cash221.jpg", prompt)
# print(output)
#
# output_2 = model_instance.img_to_text("D:\Vishva\VersionLLM\cash222.jpg", prompt)
# print(output_2)

# def get_first_image_path(root_folder):
#     image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
#     img_paths = []
#     for subdir, _, _ in os.walk(root_folder):
#         for ext in image_extensions:
#             images = glob.glob(os.path.join(subdir, ext))
#             if images:
#                 img_paths.append(images[0])
#     return img_paths

#
# root_folder = r'D:\KVBOCR\Cash_Voucher_Sucess'
# output_file = r"D:\Vishva\VersionLLM\out1.txt"
# first_image_path = get_first_image_path(root_folder)
#
# if first_image_path:
#     print("---------------------------------------------")
#     print("Total Images : ", len(first_image_path))
#     current =1
#     for path in first_image_path:
#         print("\n",path)
#         output = model_instance.img_to_text(path,prompt)
#         print(output,"\n")
#         print("Docs count: ",str(current))
#         current+=1
#         with open(output_file,  'a') as file:
#             file.write(output + '\n\n')
# else:
#     print("No images found.")
#
