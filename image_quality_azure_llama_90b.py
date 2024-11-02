# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
"""
DESCRIPTION:
    This sample demonstrates how to get a chat completions response from
    the service using a synchronous client. The sample shows how to load
    an image from a file and include it in the input chat messages.
    This sample will only work on AI models that support image input.
    Only these AI models accept the array form of `content` in the
    `UserMessage`, as shown here.

    This sample assumes the AI model is hosted on a Serverless API or
    Managed Compute endpoint. For GitHub Models or Azure OpenAI endpoints,
    the client constructor needs to be modified. See package documentation:
    https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-inference/README.md#key-concepts

USAGE:
    python sample_chat_completions_with_image_data.py

    Set these two or three environment variables before running the sample:
    1) AZURE_AI_LLAMA90B_CHAT_ENDPOINT - Your endpoint URL, in the form 
        https://<your-deployment-name>.<your-azure-region>.models.ai.azure.com
        where `your-deployment-name` is your unique AI Model deployment name, and
        `your-azure-region` is the Azure region where your model is deployed.
    2) AZURE_AI_LLAMA90B_CHAT_KEY - Your model key (a 32-character string). Keep it secret.
    3) AZURE_AI_LLAMA90B_CHAT_DEPLOYMENT_NAME - Optional. The value for the HTTP
        request header `azureml-model-deployment`.
"""

import json
import logging
import os
import time

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from PIL import Image

import utilities

# Constants
MODEL_ID = "llama-3-2-90b-vision-instruct"
DIRECTORY = "input/"
TEMPERATURE = 0
MAX_TOKENS = 512

# Read the system prompt from a file
SYSTEM_PROMPT = open("prompts/image_quality_system_prompt.txt", "r").read()

# Read the user prompt from a file
USER_PROMPT = open("prompts/image_quality_user_prompt.txt", "r").read()


def retrieve_env_vars() -> tuple:
    load_dotenv()

    try:
        endpoint = os.environ["AZURE_AI_LLAMA90B_CHAT_ENDPOINT"]
        key = os.environ["AZURE_AI_LLAMA90B_CHAT_KEY"]
    except KeyError:
        logging.error(
            "Missing environment variable 'AZURE_AI_LLAMA90B_CHAT_ENDPOINT' or 'AZURE_AI_LLAMA90B_CHAT_KEY'"
        )
        logging.error("Set them before running this sample.")
        exit()

    try:
        model_deployment = os.environ["AZURE_AI_LLAMA90B_CHAT_DEPLOYMENT_NAME"]
    except KeyError:
        logging.error(
            "Could not read optional environment variable `AZURE_AI_LLAMA90B_CHAT_DEPLOYMENT_NAME`."
        )
        logging.error("HTTP request header `azureml-model-deployment` will not be set.")
        model_deployment = None
    return endpoint, key, model_deployment


def main() -> None:
    # Retrieve the values from environment variables
    endpoint, key, model_deployment = retrieve_env_vars()

    # Initialize the client
    client = ChatCompletionsClient(
        endpoint=endpoint,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        credential=AzureKeyCredential(key),
        headers={"azureml-model-deployment": model_deployment},
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
            }

            # Send request to Azure AI Chat
            try:
                response = client.complete(payload)
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

            response_text = response.choices[0].message.content.strip()
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
            logging.info("Sleeping for 10 seconds...")
            time.sleep(10)

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
