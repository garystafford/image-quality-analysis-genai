"""
# Title: Evaluate Image Quality with Google Gemini 1.5 Pro 002 Model
# Author: Gary A. Stafford
# Date: 2024-10-21
"""

import json
import logging
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

from utilities import Utilities

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
MODEL_ID = "gemini-1.5-pro-002"
DIRECTORY = "input/"
TEMPERATURE = 0
MAX_TOKENS = 512

# Read the system prompt from a file
SYSTEM_PROMPT = open("prompts/image_quality_system_prompt.txt", "r").read()

# Read the user prompt from a file
USER_PROMPT = open("prompts/image_quality_user_prompt.txt", "r").read()


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def main() -> None:
    load_dotenv()
    genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])

    # Create the model
    generation_config = {
        "temperature": TEMPERATURE,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": MAX_TOKENS,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name=MODEL_ID,
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT,
    )

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
            # image_resize = Utilities.resize_image(image, 1120)  # max. 1120x1120 pixels
            # image = Utilities.image_to_bytes(image_resize, file_format)

            files = [
                upload_to_gemini(image_path, mime_type=f"image/{file_format}"),
            ]

            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            files[0],
                            USER_PROMPT,
                        ],
                    },
                ]
            )

            # Send request to Bedrock
            try:
                response = chat_session.send_message("INSERT_INPUT_HERE")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

            response_text = response.text.strip()
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
        with open(f"output/image_quality_google_{MODEL_ID}.json", "w") as f:
            f.write(json.dumps(scores, indent=2))
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")


if __name__ == "__main__":
    main()
