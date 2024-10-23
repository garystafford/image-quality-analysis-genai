"""
# Title: Evaluate Image Quality with Anthropic's Claude 3.5 Sonnet Model
# Author: Gary A. Stafford
# Date: 2024-10-19
"""

import json
import logging
import os
import time

from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image

from utilities import Utilities

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
MODEL_ID = "claude-3-5-sonnet-20241022"
# MODEL_ID = "claude-3-5-sonnet-20240620"
DIRECTORY = "input/"
TEMPERATURE = 0
MAX_TOKENS = 512

# Read the system prompt from a file
SYSTEM_PROMPT = open("prompts/image_quality_system_prompt.txt", "r").read()

# Read the user prompt from a file
USER_PROMPT = open("prompts/image_quality_user_prompt.txt", "r").read()


def main() -> None:
    load_dotenv()

    # Retrieve the API key from environment variables
    api_key = os.environ["ANTHROPIC_API_KEY"]

    # Initialize the Mistral client
    client = Anthropic(api_key=api_key)

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
            image_resize = Utilities.resize_image(
                image, 1568
            )  # max. 1568 pixels longest side
            image_base64 = Utilities.image_to_base64(image_resize, file_format)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{file_format}",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": USER_PROMPT,
                        },
                    ],
                },
            ]

            system_messages = SYSTEM_PROMPT

            # Send request to Anthropic
            try:
                response = client.messages.create(
                    model=MODEL_ID,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    system=system_messages,
                )
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

            response_text = response.content[0].text.strip()
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

    logging.info(json.dumps(scores, indent=2))

    # Count the scores
    logging.info(f"Scores: {Utilities.count_scores(scores)}")

    # Write the JSON results to a file
    try:
        with open(
            f"output/image_quality_anthropic_{MODEL_ID}.json", "w"
        ) as f:
            f.write(json.dumps(scores, indent=2))
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")


if __name__ == "__main__":
    main()
