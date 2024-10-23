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
        # Modify the response to include the correct image_id and time
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
        """Convert a PIL Image to bytes"""
        if isinstance(img, Image.Image):
            logging.debug("Converting PIL Image to bytes")
            buffer = io.BytesIO()
            img.save(buffer, format=file_format, quality=100)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError(f"Expected PIL Image. Got {type(img)}")

    @staticmethod
    def image_to_bytes(img: Image, file_format: str) -> bytes:
        """Convert a PIL Image to bytes"""
        if isinstance(img, Image.Image):
            logging.debug("Converting PIL Image to bytes")
            buffer = io.BytesIO()
            img.save(buffer, format=file_format, quality=100)
            return buffer.getvalue()
        else:
            raise ValueError(f"Expected PIL Image. Got {type(img)}")

    @staticmethod
    def count_scores(scores: dict) -> Counter:
        scores_counts = [score["score"] for score in scores["scores"]]
        score_counts = Counter(scores_counts)
        sorted_score_count = sorted(score_counts.items())

        return sorted_score_count
