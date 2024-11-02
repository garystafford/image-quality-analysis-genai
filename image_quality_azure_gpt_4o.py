"""
# Title: Evaluate Image Quality with Azure GPT-4o
# Author: Gary A. Stafford
# Date: 2024-10-20
"""

import json
import logging
import os
import time

import requests
from dotenv import load_dotenv
from PIL import Image

import utilities

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
MODEL_ID = "openai-gpt-4o-20240513"
DIRECTORY = "input/"
TEMPERATURE = 0
MAX_TOKENS = 512
TOP_P = 0.95


# Read the system prompt from a file
SYSTEM_PROMPT = open("prompts/image_quality_system_prompt.txt", "r").read()

# Read the user prompt from a file
USER_PROMPT = open("prompts/image_quality_user_prompt.txt", "r").read()


def retrieve_env_vars() -> tuple:
    load_dotenv()

    try:
        api_key = os.environ["AZURE_GPT4O_API_KEY"]
        endpoint = os.environ["AZURE_GPT4O_MODEL_ENDPOINT"]
    except KeyError:
        logging.error(
            "Missing environment variable 'AZURE_GPT4O_API_KEY' or 'AZURE_GPT4O_MODEL_ENDPOINT'"
        )
        logging.error("Set them before running this sample.")
        exit()

    return api_key, endpoint


def main() -> None:
    api_key, endpoint = retrieve_env_vars()

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

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
            image_base64 = utilities.image_to_base64(image, file_format)

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ]

            # Payload for the request
            payload = {
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "top_p": TOP_P,
            }

            # Send request to Bedrock
            try:
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

            response_text = (response.json())["choices"][0]["message"]["content"]
            logging.debug(f"Raw response: {response_text}")

            # Calculate time taken
            t1 = time.time()
            tt = round(t1 - t0, 2)
            logging.debug(f"Processed {filename} in {tt:.2f} seconds")

            result = utilities.truncate(response_text)
            result["image_id"] = filename
            result["model_id"] = MODEL_ID
            result["temperature"] = TEMPERATURE
            result["max_tokens"] = MAX_TOKENS
            result["time"] = tt
            scores["scores"].append(result)

            # Sleep for n seconds to avoid rate limiting
            logging.info("Sleeping for 15 seconds...")
            time.sleep(15)

    logging.info(json.dumps(scores, indent=2))

    # Count the scores
    logging.info(f"Scores: {utilities.count_scores(scores)}")

    # Write the JSON results to a file
    try:
        with open(f"output/image_quality_azure_{MODEL_ID}.json", "w") as f:
            f.write(json.dumps(scores, indent=2))
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")


if __name__ == "__main__":
    main()
