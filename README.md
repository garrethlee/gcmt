# gcmt

A simple CLI tool that uses LLMs to automatically generate meaningful & conventional commit messages from your staged changes, while also allowing you to edit the message according to your instructions.

![gcmt demo](docs/assets/demo.gif)

Using the [Hugging Face Inference API's](https://huggingface.co/docs/api-inference/rate-limits) generous free tier, you can generate commit messages without incurring additional $$$.

## Installation

```bash
pip install gcmt
```

Be sure to set your Hugging Face access token as an environment variable:

```bash
export HF_TOKEN="<your-token>"
```

You can get an access token [here](https://huggingface.co/settings/tokens) after signing up for a Hugging Face account.

## Usage

```bash
# Generates a commit message for your staged changes
gcmt

# Generates a commit message and commits them automatically, no additional input required
gcmt --auto-commit

# Choose a different model to generate the commit message. List of models [here](https://huggingface.co/docs/api-inference/supported-models)
gcmt --model_name meta-llama/Meta-Llama-3-70B-Instruct
```

## Commands

After generating a commit message, you'll be prompted with several options:

- `c`: Commit the changes with the generated message
- `cp`: Copy the generated message to clipboard
- `e "<instructions>"`: Edit the commit message according to your instructions (e.g., `e "make it more concise"`)
- `r`: Re-generate a new commit message
- `a`: Abort the commit process

> Fun Fact: All of the commit messages in this project were generated using gcmt :)