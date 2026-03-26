from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from llm.schemas import (
    AppendFileArgs,
    CreateFileArgs,
    DeleteFileArgs,
    EmptyArgs,
    GitCreateBranchArgs,
    GitDiffArgs,
    GitLogArgs,
    InspectWorkspaceArgs,
    ListFilesArgs,
    PatchFileArgs,
    ReadFileArgs,
    ReplaceInFileArgs,
    RunShellArgs,
    RunTestsArgs,
    SearchInFilesArgs,
    ShowDiffArgs,
    WriteFileArgs,
)


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], dict]
    category: str = "general"
    mutating: bool = False
    destructive: bool = False

    def prompt_line(self) -> str:
        fields = []
        for name, field in self.input_model.model_fields.items():
            required = "required" if field.is_required() else "optional"
            annotation = getattr(field.annotation, "__name__", str(field.annotation))
            fields.append(f"{name}:{annotation} ({required})")
        signature = ", ".join(fields) if fields else "no args"
        traits = [self.category]
        if self.mutating:
            traits.append("mutating")
        if self.destructive:
            traits.append("destructive")
        trait_text = ", ".join(traits)
        return f"- {self.name} [{trait_text}] ({signature}): {self.description}"


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def render_for_prompt(self) -> str:
        return "\n".join(spec.prompt_line() for spec in self._tools.values())


def build_default_registry(
    filesystem,
    search,
    shell,
    gittools,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="inspect_workspace",
            description="Build a focused repository summary and identify important files.",
            input_model=InspectWorkspaceArgs,
            handler=search.inspect_workspace,
            category="inspect",
        )
    )
    registry.register(
        ToolSpec(
            name="list_files",
            description="List files under a directory, optionally filtered by glob.",
            input_model=ListFilesArgs,
            handler=search.list_files,
            category="inspect",
        )
    )
    registry.register(
        ToolSpec(
            name="search_in_files",
            description="Search for literal text or regex matches across files.",
            input_model=SearchInFilesArgs,
            handler=search.search_in_files,
            category="inspect",
        )
    )
    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a file or a selected line range.",
            input_model=ReadFileArgs,
            handler=filesystem.read_file,
            category="read",
        )
    )
    registry.register(
        ToolSpec(
            name="write_file",
            description="Write full content to a file and return a diff.",
            input_model=WriteFileArgs,
            handler=filesystem.write_file,
            category="write",
            mutating=True,
        )
    )
    registry.register(
        ToolSpec(
            name="append_file",
            description="Append content to a file and return a diff.",
            input_model=AppendFileArgs,
            handler=filesystem.append_file,
            category="write",
            mutating=True,
        )
    )
    registry.register(
        ToolSpec(
            name="create_file",
            description="Create a new file with optional initial content.",
            input_model=CreateFileArgs,
            handler=filesystem.create_file,
            category="write",
            mutating=True,
        )
    )
    registry.register(
        ToolSpec(
            name="delete_file",
            description="Delete a single file inside the workspace.",
            input_model=DeleteFileArgs,
            handler=filesystem.delete_file,
            category="write",
            mutating=True,
            destructive=True,
        )
    )
    registry.register(
        ToolSpec(
            name="replace_in_file",
            description="Perform a literal text replacement inside a file.",
            input_model=ReplaceInFileArgs,
            handler=filesystem.replace_in_file,
            category="write",
            mutating=True,
        )
    )
    registry.register(
        ToolSpec(
            name="patch_file",
            description="Apply validated text patch operations to a file.",
            input_model=PatchFileArgs,
            handler=filesystem.patch_file,
            category="write",
            mutating=True,
        )
    )
    registry.register(
        ToolSpec(
            name="show_diff",
            description="Preview a diff without modifying the file.",
            input_model=ShowDiffArgs,
            handler=filesystem.show_diff,
            category="read",
        )
    )
    registry.register(
        ToolSpec(
            name="run_shell",
            description="Run a guarded shell command inside the workspace.",
            input_model=RunShellArgs,
            handler=shell.run_shell,
            category="execute",
        )
    )
    registry.register(
        ToolSpec(
            name="run_tests",
            description="Run a targeted test, lint, typecheck, or build command.",
            input_model=RunTestsArgs,
            handler=shell.run_tests,
            category="verify",
        )
    )
    registry.register(
        ToolSpec(
            name="git_status",
            description="Run git status --short.",
            input_model=EmptyArgs,
            handler=gittools.git_status,
            category="git",
        )
    )
    registry.register(
        ToolSpec(
            name="git_diff",
            description="Show git diff output.",
            input_model=GitDiffArgs,
            handler=gittools.git_diff,
            category="git",
        )
    )
    registry.register(
        ToolSpec(
            name="git_log",
            description="Show recent git history.",
            input_model=GitLogArgs,
            handler=gittools.git_log,
            category="git",
        )
    )
    registry.register(
        ToolSpec(
            name="git_create_branch",
            description="Create and switch to a new local branch.",
            input_model=GitCreateBranchArgs,
            handler=gittools.git_create_branch,
            category="git",
            mutating=True,
        )
    )
    return registry
