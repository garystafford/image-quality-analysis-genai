"""
# Title: Utilities for image quality assessments
# Author: Gary A. Stafford
# Date: 2024-10-14
"""

import base64
from collections import Counter
import io
import json
import logging

from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Utilities:
    @staticmethod
    def truncate(response: str) -> dict:
        """
        Truncate and parse a JSON response string.

        This function modifies the input response string to ensure it starts and ends with
        curly braces, then attempts to parse it as a JSON object. If the parsing fails,
        it returns a dictionary with an error explanation and a score of -1.

        Args:
            response (str): The raw response string to be truncated and parsed.

        Returns:
            dict: The parsed JSON object as a dictionary. If parsing fails, returns a
                  dictionary with an error explanation and a score of -1.
        """
        try:
            if not response.startswith("{") and response.count("{") > 0:
                delimiter = "{"
                response = "{" + response.split(delimiter, 1)[1]
                print(f"left_fix: {response}")
            if not response.endswith("}") and response.count("}") > 0:
                delimiter = "}"
                response = response.split(delimiter, 1)[0] + "}"
                print(f"right_fix: {response}")
            print(f"final_raw: {response}")
            result = json.loads(response)
            logging.debug(f"Response (JSON): {json.dumps(result, indent=2)}")
        except json.JSONDecodeError:
            logging.error(f"Error parsing JSON. Raw response: {response}")
            result = {}
            result["explanation"] = "Error parsing JSON response."
            result["score"] = -1

        return result

    @staticmethod
    def resize_image(img: Image, max_pixels: int) -> Image:
        """
        Resize an image to ensure its largest dimension does not exceed a specified maximum number of pixels.

        Args:
            img (Image): The input image to be resized.
            max_pixels (int): The maximum number of pixels for the largest dimension of the image.

        Returns:
            Image: The resized image, if resizing was necessary; otherwise, the original image.
        """
        width, height = img.size
        if max(width, height) > max_pixels:
            scale_factor = max_pixels / max(width, height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            logging.info(
                f"Resizing image from {width}x{height} to {new_size[0]}x{new_size[1]}"
            )
            img = img.resize(new_size, Image.LANCZOS)

        return img

    @staticmethod
    def image_to_base64(img: Image, file_format: str) -> str:
        """
        Convert a PIL Image to a base64 encoded string.

        Args:
            img (Image): The PIL Image to be converted.
            file_format (str): The format to save the image in (e.g., 'JPEG', 'PNG').

        Returns:
            str: The base64 encoded string representation of the image.

        Raises:
            ValueError: If the provided img is not an instance of PIL.Image.
        """
        if isinstance(img, Image.Image):
            logging.debug("Converting PIL Image to bytes")
            buffer = io.BytesIO()
            img.save(buffer, format=file_format, quality=100)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError(f"Expected PIL Image. Got {type(img)}")

    @staticmethod
    def image_to_bytes(img: Image, file_format: str) -> bytes:
        """
        Convert a PIL Image to bytes.

        Args:
            img (Image): The PIL Image to convert.
            file_format (str): The format to save the image in (e.g., 'JPEG', 'PNG').

        Returns:
            bytes: The image data in bytes.

        Raises:
            ValueError: If the provided img is not an instance of PIL Image.
        """
        if isinstance(img, Image.Image):
            logging.debug("Converting PIL Image to bytes")
            buffer = io.BytesIO()
            img.save(buffer, format=file_format, quality=100)
            return buffer.getvalue()
        else:
            raise ValueError(f"Expected PIL Image. Got {type(img)}")

    @staticmethod
    def count_scores(scores: dict) -> Counter:
        """
        Count the occurrences of each score in the provided dictionary and return them sorted.

        Args:
            scores (dict): A dictionary containing a list of score dictionaries under the key "scores".
                           Each score dictionary should have a "score" key.

        Returns:
            Counter: A Counter object with the counts of each score, sorted by score.
        """
        scores_counts = [score["score"] for score in scores["scores"]]
        score_counts = Counter(scores_counts)
        sorted_score_count = sorted(score_counts.items())

        return sorted_score_count
