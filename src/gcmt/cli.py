import os
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pyperclip
import typer
from huggingface_hub import InferenceClient
from rich import print
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

MAX_TOTAL_TOKENS = 8192
MAX_OUTPUT_TOKENS = 75
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - MAX_OUTPUT_TOKENS

EDIT_PROMPT = re.compile(r'"(.*)"')

app = typer.Typer(help="CLI tool for generating commit messages using LLMs")
console = Console()
api_key = os.getenv("HF_TOKEN")

if not api_key:
    console.print(
        Panel.fit(
            "[red]HF_TOKEN is not set.[/red]\nTo create an access token, go to [link]https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-tokens[/link]",
            title="Error",
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
    user_prompt: Optional[str] = None,
    **kwargs,
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

    content = prompt.format(diff_summaries=diff_summaries)

    # Add user prompt if provided
    if user_prompt:
        if len(content) + len(user_prompt) > MAX_INPUT_TOKENS:
            # Add ``` to the end in case of truncation
            # TODO: Handle this better
            content = content[: MAX_INPUT_TOKENS - len(user_prompt) - 3] + "```"
        content += f"\n\n Make sure to include the following in your commit message: {user_prompt}"

    resp = client.chat_completion(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content[:MAX_INPUT_TOKENS],
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        **kwargs,
    )

    return resp.choices[0].message.content


def edit_commit_message(
    commit_message: str,
    model_name: str,
    instructions: str,
    prompt: str = (
        "Edit the following commit message: {commit_message} "
        "according to the following instructions: {instructions}. "
        "Output the edited commit message and nothing else."
    ),
    **kwargs,
) -> str:
    """Edit an existing commit message according to provided instructions.

    Args:
        commit_message: The original commit message to edit
        model_name: The name of the model to use for editing the commit message
        instructions: Instructions on how to edit the commit message
        prompt: Template string for the prompt to send to the LLM. Should contain {commit_message}
               and {instructions} placeholders which will be replaced with the actual values
        **kwargs: Additional keyword arguments to pass to the chat completion API

    Returns:
        str: The edited commit message
    """
    resp = client.chat_completion(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt.format(
                    commit_message=commit_message, instructions=instructions
                )[:MAX_INPUT_TOKENS],
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        **kwargs,
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
    """Generate commit messages for staged changes using the Hugging Face Inference API."""
    git_diff_filenames = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"]
    ).decode("utf-8")

    if not git_diff_filenames:
        console.print("[yellow]No changes to commit[/yellow]")
        return

    diffs = [
        subprocess.check_output(["git", "diff", "--cached", "--", file]).decode("utf-8")
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
            commit_message, title="Generated Commit Message", border_style="green"
        )
    )

    if autocommit:
        console.print("[bold green]Auto-committing changes...[/bold green]")
        subprocess.run(commit_command, shell=True)
        return

    while True:
        choice = (
            typer.prompt(
                "\nAction [(c)ommit, (cp) copy to clipboard, (e)dit, (r)e-generate, (a)bort]",
                show_choices=True,
                show_default=True,
                default="c",
            )
            .lower()
            .strip()
        )

        if choice == "c":
            console.print("[bold green]Committing changes...[/bold green]")
            subprocess.run(commit_command, shell=True)
            break
        elif choice == "cp":
            pyperclip.copy(commit_message)
            console.print("[green]✓[/green] Commit message copied to clipboard!")
            break
        elif choice == "r":
            with console.status("[bold green]Re-generating commit message..."):
                commit_message = generate_commit_message(
                    diff_summaries, model_name, seed=random.randint(0, 2**32 - 1)
                )
            console.print(
                Panel.fit(
                    commit_message,
                    title="Re-generated Commit Message",
                    border_style="green",
                )
            )
        elif "e" in choice:
            user_prompt = EDIT_PROMPT.search(choice)
            if user_prompt:
                user_prompt = user_prompt.group(1)
            else:
                console.print('[yellow]Syntax: e "<instructions>"[/yellow]')
                continue

            with console.status("[bold green]Re-generating commit message..."):
                commit_message = edit_commit_message(
                    commit_message,
                    model_name,
                    instructions=user_prompt,
                )
            # Always show the generated message first
            console.print(
                Panel.fit(
                    commit_message,
                    title="Generated Commit Message",
                    border_style="green",
                )
            )
        elif choice == "a":
            console.print("[yellow]Commit aborted[/yellow]")
            raise typer.Abort()
