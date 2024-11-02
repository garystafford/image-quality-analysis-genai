# Quantitative and Qualitative Image Analysis Using Nine Different Multimodal Generative AI Vision Models

Learn to analyze image quality using state-of-the-art vision models from Anthropic, Google, Meta, Microsoft, Mistral, NVIDIA, and OpenAI.

## Installation Instructions

### 1. Setup a Python 3 Virtual Environment

Create a Python 3 virtual environment and install all required Python packages. Instructions are for Windows and Mac. I'm currently running Python 3.12.x.

#### Windows

```bat
python --version

python -m pip install virtualenv -U
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -U
```

#### Mac

```sh
python --version

python -m pip install virtualenv -U # --break-system-package
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -U
```

### 2. Setup the `.env` File

Rename the `env_template` file to `.env`. Add your sensitive credentials to the `dotenv` `.env` file for various hosting providers.

```ini
ANTHROPIC_API_KEY=""

AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_SESSION_TOKEN=""

AZURE_AI_LLAMA11B_CHAT_DEPLOYMENT_NAME=""
AZURE_AI_LLAMA11B_CHAT_ENDPOINT=""
AZURE_AI_LLAMA11B_CHAT_KEY=""

AZURE_AI_LLAMA90B_CHAT_DEPLOYMENT_NAME=""
AZURE_AI_LLAMA90B_CHAT_ENDPOINT=""
AZURE_AI_LLAMA90B_CHAT_KEY=""

AZURE_AI_PHI_CHAT_DEPLOYMENT_NAME=""
AZURE_AI_PHI_CHAT_ENDPOINT=""
AZURE_AI_PHI_CHAT_KEY=""

AZURE_GPT4O_API_KEY=""
AZURE_GPT4O_MODEL_ENDPOINT=""

GOOGLE_GEMINI_API_KEY=""

MISTRAL_API_KEY=""

NVIDIA_API_KEY=""
```

### 3. Add Images

Add images to be analyzed to the `input/` directory.

### 4. Run Python Script(s)

Run any of the 11 scripts, depending on the model and hosting platform.

```sh
python image_quality_anthropic_claude.py
```

Check the `output/` directory for results.

### Azure

For Azure AI Studio, you may need to log in first.

Install the Azure CLI for Windows: <https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli>

Install the Azure CLI for Mac:

```sh
brew update && brew install azure-cli
```

Login:

```sh
az --version

az login
```

### Output Format

```json
{
  "scores": [
    {
      "score": 2,
      "explanation": "The image is perfectly focused and sharp throughout. The exposure is ideal with excellent dynamic range, and there is minimal to no visible noise or grain. The composition and framing are excellent, capturing the entire bathroom area effectively. The resolution is high with crisp details, and the color reproduction is accurate and vibrant. The lighting is well-balanced, and there are no visible artifacts or distortions. The color balance and white point are correct.",
      "image_id": "image_01.jpg",
      "model_id": "openai-gpt-4o-2024-05-13",
      "temperature": 0.0,
      "max_tokens": 1024,
      "time": 3.45
    },
    {
      "score": 1,
      "explanation": "The image is somewhat sharp but not perfectly focused. The exposure is slightly underexposed, and there is noticeable but not excessive noise. The composition is decent, with the subject well-framed, but there is room for improvement. The resolution is adequate for general viewing, and the color reproduction is acceptable. There are no visible artifacts or distortions, and the color balance is mostly correct, though there are minor issues.",
      "image_id": "image_02.jpg",
      "model_id": "openai-gpt-4o-2024-05-13",
      "temperature": 0.0,
      "max_tokens": 1024,
      "time": 4.41
    },
    {
      "score": 0,
      "explanation": "The image is extremely blurry and out of focus, particularly in the background. The exposure is poor, with the fire being overexposed and the surrounding area underexposed. There is noticeable noise and grain, and the composition is not ideal. The resolution appears low, and there are significant color issues due to the lighting conditions. Overall, the image quality is poor and does not meet the criteria for higher scores.",
      "image_id": "image_03.jpg",
      "model_id": "openai-gpt-4o-2024-05-13",
      "temperature": 0.0,
      "max_tokens": 1024,
      "time": 3.14
    },
    {
      "score": 0,
      "explanation": "The image is of poor quality due to several factors: it is overexposed, resulting in loss of detail in the white areas of the birds. The composition is poor, with the fence obstructing the view and creating a distracting pattern. The image also appears to have noise and grain, and the overall resolution is low, leading to a lack of crisp details. Additionally, the color balance is off, with a significant color shift. These issues make it difficult to evaluate the quality of the image based on the provided criteria.",
      "image_id": "image_04.jpg",
      "model_id": "openai-gpt-4o-2024-05-13",
      "temperature": 0.0,
      "max_tokens": 1024,
      "time": 2.59
    }
  ]
}
```
