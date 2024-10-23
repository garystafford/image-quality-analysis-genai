# Evaluating Image Quality Using SOTA Multimodal Generative AI Vision Models

Learn to employ a variety of open- and closed-weight vision models from model builders Anthropic, Google, Meta AI, Microsoft, Mistral, NVIDIA, and OpenAI to analyze image quality.

## 1. Setup a Python 3 Virtual Environment

Create a Python 3 virtual environment and install all required Python packages. Instructions are for Windows and Mac. I'm currently running Python 3.12.7.

### Windows

```bat
python --version

python -m pip install virtualenv -U
python -m venv .venv
.venv\Scripts\activate

python -m pip install -r requirements.txt -U
```

### Mac

```sh
python --version

python -m pip install virtualenv -U
python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt -U
```

## 2. Setup the `.env` File

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

## 3. Add Images

Add images to be analyzed to the `input/` directory.

## 4. Run Scipt(s)

Run any of the 11 scripts, depending on the model and hosting platform.

```sh
python image_quality_anthropic_claude.py
```

Check the `output/` directory for results.

## Azure

For Azure AI Studio, you may need to login first.

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
