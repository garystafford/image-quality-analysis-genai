"""
# Title: Evaluate Image Quality with NVIDIA Neva-22b
# Author: Gary A. Stafford
# Date: 2024-10-22
"""

import json
import logging
import os
import time
import requests

from dotenv import load_dotenv
from PIL import Image

from utilities import Utilities

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
MODEL_ID = "neva-22b"
DIRECTORY = "input/"
TEMPERATURE = 0.000001  # can't be 0 - throws an error
MAX_TOKENS = 512

# Read the system prompt from a file
SYSTEM_PROMPT = open("prompts/image_quality_system_prompt.txt", "r").read()

# Read the user prompt from a file
USER_PROMPT = open("prompts/image_quality_user_prompt.txt", "r").read()


def main() -> None:
    load_dotenv()

    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
    gemini_api_key = os.environ["NVIDIA_API_KEY"]
    stream = False

    headers = {
        "Authorization": f"Bearer {gemini_api_key}",
        "Accept": "text/event-stream" if stream else "application/json",
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
            image_resize = Utilities.resize_image(image, 1536)  # max. 180_000
            image_base64 = Utilities.image_to_base64(image_resize, file_format)

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f'{USER_PROMPT} <img src="data:image/{file_format};base64,{image_base64}" />',
                    },
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": 0.70,
                "seed": 0,
            }

            # Send request to the API
            try:
                response = requests.post(invoke_url, headers=headers, json=payload)
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

            if stream:
                for line in response.iter_lines():
                    if line:
                        logging.debug(line.decode("utf-8"))
            else:
                logging.info(response.json())

            response_text = (response.json())["choices"][0]["message"]["content"]
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

            # Sleep for n seconds to avoid rate limiting
            # logging.info("Sleeping for 25 seconds...")
            # time.sleep(25)

    logging.info(json.dumps(scores, indent=2))

    # Count the scores
    logging.info(f"Scores: {Utilities.count_scores(scores)}")

    # Write the JSON results to a file
    try:
        with open(f"output/image_quality_nvidia_{MODEL_ID}.json", "w") as f:
            f.write(json.dumps(scores, indent=2))
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")


if __name__ == "__main__":
    main()
