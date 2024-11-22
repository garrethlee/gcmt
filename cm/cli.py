import typer
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient
from typing_extensions import Annotated
from typing import List
import pyperclip
from rich import print
from rich.panel import Panel
from rich.console import Console

MAX_TOTAL_TOKENS = 8192
MAX_OUTPUT_TOKENS = 75
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - MAX_OUTPUT_TOKENS

app = typer.Typer(help="CLI tool for generating commit messages using LLMs")
console = Console()
api_key = os.getenv("HF_TOKEN")

if not api_key:
    console.print(
        Panel.fit(
            "[red]HF_TOKEN is not set.[/red]\nTo create an access token, go to [link]https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-tokens[/link]",
            title="Error"
        )
    )
    exit(1)

client = InferenceClient(api_key=api_key)


def batch_diffs(diffs: List[str]) -> List[str]:
    """Takes a list of git diffs and batches them together while ensuring
    each batch stays under MAX_INPUT_TOKENS in length. If a single diff is larger than
    MAX_INPUT_TOKENS, it will be truncated.

    Args:
        diffs: A list of strings containing git diffs to be batched

    Returns:
        List[str]: A list of batched diffs, where each batch is a string containing
                  one or more diffs joined by newlines and staying under MAX_INPUT_TOKENS
                  in length
    """
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
        typer.Option(
            help="Automatically commit the changes after generating the commit message"
        ),
    ] = False,
):
    """Generate commit messages for staged changes using LLMs."""
    git_diff_filenames = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"]
    ).decode("utf-8")

    if not git_diff_filenames:
        console.print("[yellow]No changes to commit[/yellow]")
        return

    with console.status("[bold green]Analyzing changes..."):
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
    with console.status("[bold green]Generating commit message..."):
        commit_message = generate_commit_message(diff_summaries, model_name)
        commit_command = f"git commit -m '{commit_message}'"

    # Always show the generated message first
    console.print(
        Panel.fit(
            commit_message,
            title="Generated Commit Message",
            border_style="green"
        )
    )

    if autocommit:
        console.print("[bold green]Auto-committing changes...[/bold green]")
        subprocess.run(commit_command, shell=True)
        return

    while True:
        choice = typer.prompt(
            "\nAction",
            type=click.Choice(["c", "cp", "a"], case_sensitive=False),
            show_choices=True,
            show_default=True,
            default="c",
        )

        if choice == "c":
            console.print("[bold green]Committing changes...[/bold green]")
            subprocess.run(commit_command, shell=True)
            break
        elif choice == "cp":
            pyperclip.copy(commit_message)
            console.print("[green]✓[/green] Commit message copied to clipboard!")
            break
        elif choice == "a":
            console.print("[yellow]Commit aborted[/yellow]")
            raise typer.Abort()


if __name__ == "__main__":
    app()
