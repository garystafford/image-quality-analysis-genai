"""
# Title: Evaluate Image Quality with Amazon Bedrock Meta Llama 3.2 11b Vision Instruct Model
# Author: Gary A. Stafford
# Date: 2024-10-13
"""

import json
import logging
import os
import time

import boto3
from PIL import Image

from utilities import Utilities

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
MODEL_ID = "us.meta.llama3-2-11b-instruct-v1:0"
DIRECTORY = "input/"
TEMPERATURE = 0
MAX_TOKENS = 512

# Read the system prompt from a file
SYSTEM_PROMPT = open("prompts/image_quality_system_prompt.txt", "r").read()

# Read the user prompt from a file
USER_PROMPT = open("prompts/image_quality_user_prompt.txt", "r").read()


def main() -> None:
    # Create a Bedrock Runtime client
    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Base inference parameters to use
    inference_config = {"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS}

    # Initialize the scores dictionary
    scores = {"scores": []}

    # Iterate over all combined images in the directory
    for filename in sorted(os.listdir(DIRECTORY)):
        t0 = time.time()

        logging.info(f"Evaluation for {filename}")

        if filename.endswith((".jpeg", ".jpg", ".png")):
            image_path = os.path.join(DIRECTORY, filename)
            image = Image.open(image_path)
            file_format = "jpeg" if image.format.lower() in ["jpg", "jpeg"] else "png"
            image_resize = Utilities.resize_image(image, 1120)  # max. 1120x1120 pixels
            image = Utilities.image_to_bytes(image_resize, file_format)

            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": file_format, "source": {"bytes": image}}},
                        {"text": USER_PROMPT},
                    ],
                }
            ]

            system_messages = [
                {
                    "text": SYSTEM_PROMPT,
                }
            ]

            # Send request to Bedrock
            try:
                response = bedrock_runtime.converse(
                    modelId=MODEL_ID,
                    system=system_messages,
                    messages=user_messages,
                    inferenceConfig=inference_config,
                )
            except bedrock_runtime.exceptions.ValidationException as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

            response_text = response["output"]["message"]["content"][0]["text"].strip()
            logging.debug(f"Raw response: {response_text}")

            # Calculate time taken
            t1 = time.time()
            tt = round(t1 - t0, 2)
            logging.debug(f"Processed {filename} in {tt:.2f} seconds")

            result = Utilities.truncate(response_text)
            result["image_id"] = filename
            result["model_id"] = MODEL_ID
            result["temperature"] = TEMPERATURE
            result["max_tokens"] = MAX_TOKENS
            result["time"] = tt
            scores["scores"].append(result)

            # Sleep for n seconds to avoid rate limiting from Bedrock Isengard account
            logging.info("Sleeping for 25 seconds...")
            time.sleep(25)

    logging.info(json.dumps(scores, indent=2))

    # Count the scores
    logging.info(f"Scores: {Utilities.count_scores(scores)}")

    # Write the JSON results to a file
    try:
        with open("output/image_quality_bedrock_llama3-2-11b-instruct.json", "w") as f:
            f.write(json.dumps(scores, indent=2))
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")


if __name__ == "__main__":
    main()
