import typer
import click
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient
from typing_extensions import Annotated
from typing import List
import pyperclip



MAX_TOTAL_TOKENS = 8192
MAX_OUTPUT_TOKENS = 75
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - MAX_OUTPUT_TOKENS

app = typer.Typer()
api_key = os.getenv("HF_TOKEN")

if not api_key:
    typer.echo(
        "HF_TOKEN is not set. To create an access token, go to https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-tokens"
    )
    exit(1)

client = InferenceClient(api_key=api_key)


def batch_diffs(diffs: List[str]) -> List[str]:
    """Batch multiple git diffs up to MAX_INPUT_TOKENS"""
    batched_diffs = []
    current_batch = []
    for diff in diffs:
        if len(diff) > MAX_INPUT_TOKENS:
            batched_diffs.append(diff[:MAX_INPUT_TOKENS])
        elif len(current_batch) + len(diff) > MAX_INPUT_TOKENS:
            batched_diffs.append("\n".join(current_batch))
            current_batch = [diff]
        else:
            current_batch.append(diff)
    batched_diffs.append("\n".join(current_batch))
    return batched_diffs


def summarize_changes_in_file(
    changes: str,
    model_name: str,
    prompt: str = "Summarize the changes in this file's git diff. Output a one-line summary and nothing else. Here is the git diff: {diff}",
) -> str:
    """Summarize the changes in a single git-tracked file

    Args:
        changes: The git diff to summarize
        model_name: The name of the model to use for summarizing the changes
        prompt: Template string for the prompt to send to the LLM. Should contain {diff} placeholder
                which will be replaced with the file's git diff.

    Returns:
        str: The LLM-generated summary of changes in the file
    """
    resp = client.chat_completion(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt.format(diff=changes)[:MAX_INPUT_TOKENS],
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    return resp.choices[0].message.content


def generate_commit_message(
    diff_summaries: List[str],
    model_name: str,
    prompt: str = (
        "Generate a one-line commit message for the changes based on the following changes. "
        "Use conventional commit messages, which has the following structure: "
        "<type>[optional scope]:\n\n<description>\n\n[optional body]\n\n[optional footer(s)]. "
        "Examples of types: fix: and feat: are allowed, chore:, ci:, docs:, style:, refactor:, perf:, test:"
        "Output the commit message and nothing else. "
        "Here are the changes: {diff_summaries}"
    ),
) -> str:
    """Generate a commit message based on the summaries of the changes in the files.

    Args:
        diff_summaries: List of summaries of the changes in the files
        model_name: The name of the model to use for generating the commit message
        prompt: The prompt to use for generating the commit message

    Returns:
        str: The generated commit message
    """

    diff_summaries = "\n".join(diff_summaries)

    resp = client.chat_completion(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt.format(diff_summaries=diff_summaries)[
                    :MAX_INPUT_TOKENS
                ],
            },
        ],
        # response_format={
        #     "type": "json",
        #     "value": {
        #         "properties": {
        #             "message": {"type": "string"},
        #         },
        #     },
        # },
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    return resp.choices[0].message.content


@app.command()
def commit(
    model_name: Annotated[
        str,
        typer.Option(
            help="The name of the model to use for generating the commit message"
        ),
    ] = "meta-llama/Meta-Llama-3-70B-Instruct",
    autocommit: Annotated[
        bool,
        typer.Option(help="Automatically commit the changes after generating the commit message"),
    ] = False,
):
    git_diff_filenames = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"]
    ).decode("utf-8")

    if not git_diff_filenames:
        typer.echo("No changes to commit")
        return

    diffs = [
        subprocess.check_output(["git", "diff", "--cached", file]).decode("utf-8")
        for file in git_diff_filenames.splitlines()
    ]
    batched_diffs = batch_diffs(diffs)

    with ThreadPoolExecutor() as executor:
        diff_summaries = list(
            executor.map(
                lambda changes: summarize_changes_in_file(changes, model_name),
                batched_diffs,
            )
        )

    # Generate commit message based on summaries
    commit_message = generate_commit_message(diff_summaries, model_name)
    commit_command = f"git commit -m '{commit_message}'"

    # Always show the generated message first
    typer.echo(f'\nGenerated commit message:\n"{commit_message}"\n')

    if autocommit:
        typer.echo("Auto-committing changes...")
        subprocess.run(commit_command, shell=True)
        return

    # Present options to the user
    choices = ["(c)ommit", "(cp) copy to clipboard", "(a)bort"]


    while True:
        choice = typer.prompt(
            "Action ([c]ommit / [cp] copy to clipboard / [a]bort)",
        )

        if choice in ["(c)ommit", "c"]:
            # subprocess.run(commit_command, shell=True)
            typer.echo("Committed!")
            break
        elif choice in ["(cp) copy to clipboard", "cp"]:
            pyperclip.copy(commit_message)
            typer.echo("Commit message copied to clipboard!")
            break
        elif choice in ["(a)bort", "a"]:
            typer.echo("Commit aborted")
            raise typer.Abort()
        else:
            typer.echo("Invalid choice. Choose one of the options above.")

if __name__ == "__main__":
    app()
