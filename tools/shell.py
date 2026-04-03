from __future__ import annotations

import json
import py_compile
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
import textwrap

from config.settings import AppConfig
from llm.schemas import RunShellArgs, RunTestsArgs
from runtime.workspace import WorkspaceManager
from tools.safety import SafetyManager


ZERO_TEST_OUTPUT_PATTERNS = (
    re.compile(r"\bran\s+0\s+tests?\b", re.IGNORECASE),
    re.compile(r"\bcollected\s+0\s+items\b", re.IGNORECASE),
    re.compile(r"\bno\s+tests\s+ran\b", re.IGNORECASE),
)
RUNTIME_PYTHON_PREFIX = re.compile(
    r"^(?P<prefix>(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s]+)\s+)*)"
    r"(?P<python>python(?:3)?)\b"
)


class ShellTools:
    def __init__(
        self,
        config: AppConfig,
        workspace: WorkspaceManager,
        safety: SafetyManager,
    ):
        self.config = config
        self.workspace = workspace
        self.safety = safety

    def run_shell(self, args: RunShellArgs) -> dict:
        return self._run_command(args.command, args.cwd, args.timeout)

    def run_tests(self, args: RunTestsArgs) -> dict:
        if args.command.startswith("internal:"):
            return self._run_internal_validation(args.command, args.cwd)
        result = self._run_command(args.command, args.cwd, args.timeout)
        insufficient_reason = self._insufficient_test_execution_reason(
            args.command,
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
        )
        if result.get("success") and insufficient_reason:
            stderr = str(result.get("stderr") or "").strip()
            result["success"] = False
            result["message"] = insufficient_reason
            result["stderr"] = (
                f"{stderr}\n{insufficient_reason}".strip()
                if stderr
                else insufficient_reason
            )
            result["insufficient_verification"] = True
            return result
        result["message"] = (
            f"Validation command exited with {result['exit_code']}."
            if "exit_code" in result
            else result["message"]
        )
        return result

    def _run_internal_validation(self, command: str, cwd: str) -> dict:
        working_dir = self.workspace.resolve_directory(cwd)
        if not working_dir.exists():
            return {
                "success": False,
                "message": f"Working directory does not exist: {working_dir}",
                "risk_level": "low",
                "blocked": True,
                "command": command,
            }

        kind, _, payload = command.partition(":")
        kind, _, payload = payload.partition(":")
        try:
            relative_paths = json.loads(payload or "[]")
        except json.JSONDecodeError:
            return {
                "success": False,
                "message": "Internal validation payload could not be decoded.",
                "risk_level": "low",
                "command": command,
            }

        if kind == "python_syntax":
            return self._run_python_syntax_validation(command, working_dir, relative_paths)
        if kind == "python_cli_smoke":
            return self._run_python_cli_smoke_validation(command, working_dir, relative_paths)
        if kind == "web_artifact":
            return self._run_web_artifact_validation(command, working_dir, relative_paths)
        if kind == "web_runtime_smoke":
            return self._run_web_runtime_smoke_validation(command, working_dir, relative_paths)
        if kind == "html_refs":
            return self._run_html_reference_validation(command, working_dir, relative_paths)
        return {
            "success": False,
            "message": f"Unknown internal validation kind: {kind}",
            "risk_level": "low",
            "command": command,
        }

    def _run_command(self, command: str, cwd: str, timeout: int | None) -> dict:
        assessment = self.safety.assess_shell_command(command)
        if not assessment.allowed:
            return {
                "success": False,
                "message": "; ".join(assessment.reasons),
                "risk_level": assessment.risk_level,
                "blocked": True,
                "command": command,
            }

        working_dir = self.workspace.resolve_directory(cwd)
        if not working_dir.exists():
            return {
                "success": False,
                "message": f"Working directory does not exist: {working_dir}",
                "risk_level": assessment.risk_level,
                "blocked": True,
                "command": command,
            }
        if self.config.dry_run:
            return {
                "success": True,
                "message": f"Dry run: would execute '{command}' in {working_dir}",
                "risk_level": assessment.risk_level,
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "command": command,
            }

        effective_timeout = timeout or self.config.shell_timeout
        resolved_command = self._resolve_runtime_command(command)
        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", resolved_command],
                cwd=working_dir,
                text=True,
                capture_output=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"Command timed out after {effective_timeout} seconds.",
                "risk_level": assessment.risk_level,
                "timeout": True,
                "command": command,
            }

        stdout = completed.stdout[-self.config.max_read_chars :]
        stderr = completed.stderr[-self.config.max_read_chars :]
        return {
            "success": completed.returncode == 0,
            "message": f"Command exited with {completed.returncode}.",
            "risk_level": assessment.risk_level,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": completed.returncode,
            "command": command,
            "resolved_command": resolved_command,
        }

    def _insufficient_test_execution_reason(
        self,
        command: str,
        *,
        stdout: str,
        stderr: str,
    ) -> str | None:
        lowered = str(command or "").lower()
        if not any(token in lowered for token in ("pytest", "unittest", "go test", "cargo test")):
            return None
        combined = "\n".join(part for part in (stdout, stderr) if part).strip()
        if not combined:
            return None
        if any(pattern.search(combined) for pattern in ZERO_TEST_OUTPUT_PATTERNS):
            return "Validation command did not execute any tests."
        return None

    def _resolve_runtime_command(self, command: str) -> str:
        raw_command = str(command or "").strip()
        if not raw_command:
            return raw_command
        match = RUNTIME_PYTHON_PREFIX.match(raw_command)
        if match is None:
            return raw_command

        prefix = match.group("prefix") or ""
        suffix = raw_command[match.end("python") :]
        runtime_python = shlex.quote(sys.executable)
        return f"{prefix}{runtime_python}{suffix}"

    def _run_python_syntax_validation(self, command: str, working_dir: Path, relative_paths: list[str]) -> dict:
        failures: list[str] = []
        checked = 0
        for relative_path in relative_paths:
            target = self.workspace.resolve_path(working_dir / relative_path)
            if not target.exists():
                failures.append(f"Missing file: {relative_path}")
                continue
            try:
                py_compile.compile(str(target), doraise=True)
                checked += 1
            except py_compile.PyCompileError as exc:
                failures.append(str(exc))

        success = not failures and checked > 0
        return {
            "success": success,
            "message": "Validation command exited with 0." if success else "Validation command exited with 1.",
            "risk_level": "low",
            "stdout": f"Checked {checked} Python file(s).",
            "stderr": "\n".join(failures),
            "exit_code": 0 if success else 1,
            "command": command,
        }

    def _run_python_cli_smoke_validation(self, command: str, working_dir: Path, relative_paths: list[str]) -> dict:
        failures: list[str] = []
        outputs: list[str] = []
        checked = 0
        effective_timeout = min(max(int(self.config.shell_timeout), 1), 8)

        for relative_path in relative_paths[:2]:
            target = self.workspace.resolve_path(working_dir / relative_path)
            if not target.exists():
                failures.append(f"Missing file: {relative_path}")
                continue
            if target.suffix.lower() != ".py":
                failures.append(f"Not a Python file: {relative_path}")
                continue

            try:
                completed = subprocess.run(
                    [sys.executable, str(target)],
                    cwd=working_dir,
                    text=True,
                    capture_output=True,
                    input=self._default_python_smoke_input(target),
                    timeout=effective_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = str(exc.stdout or "")[-self.config.max_read_chars :]
                stderr = str(exc.stderr or "")[-self.config.max_read_chars :]
                return {
                    "success": False,
                    "message": f"Validation command timed out after {effective_timeout} seconds.",
                    "risk_level": "low",
                    "stdout": stdout,
                    "stderr": stderr,
                    "timeout": True,
                    "command": command,
                }

            checked += 1
            stdout = completed.stdout[-self.config.max_read_chars :]
            stderr = completed.stderr[-self.config.max_read_chars :]
            outputs.append(f"$ {relative_path}\n{stdout}".strip())
            if completed.returncode != 0:
                failure_text = stderr or stdout or f"Exited with {completed.returncode}."
                failures.append(f"{relative_path}: {failure_text}".strip())

        success = not failures and checked > 0
        return {
            "success": success,
            "message": "Validation command exited with 0." if success else "Validation command exited with 1.",
            "risk_level": "low",
            "stdout": "\n\n".join(part for part in outputs if part),
            "stderr": "\n".join(failures),
            "exit_code": 0 if success else 1,
            "command": command,
        }

    def _run_html_reference_validation(self, command: str, working_dir: Path, relative_paths: list[str]) -> dict:
        parser = _HTMLReferenceParser()
        missing_refs: list[str] = []
        checked = 0
        for relative_path in relative_paths:
            target = self.workspace.resolve_path(working_dir / relative_path)
            if not target.exists():
                missing_refs.append(f"Missing HTML file: {relative_path}")
                continue
            parser.reset_refs()
            parser.feed(target.read_text(encoding="utf-8"))
            checked += 1
            for reference in parser.references:
                resolved = (target.parent / reference).resolve()
                if not resolved.exists():
                    missing_refs.append(f"{relative_path} -> {reference}")

        success = not missing_refs and checked > 0
        return {
            "success": success,
            "message": "Validation command exited with 0." if success else "Validation command exited with 1.",
            "risk_level": "low",
            "stdout": f"Checked {checked} HTML file(s).",
            "stderr": "\n".join(missing_refs),
            "exit_code": 0 if success else 1,
            "command": command,
        }

    def _run_web_artifact_validation(self, command: str, working_dir: Path, payload_items: list) -> dict:
        failures: list[str] = []
        summaries: list[str] = []
        checked = 0
        node_binary = shutil.which("node")

        for item in payload_items:
            descriptor = item if isinstance(item, dict) else {"path": item}
            relative_path = str(descriptor.get("path") or "").strip()
            expected_features = [
                str(feature or "").strip()
                for feature in descriptor.get("expected_features", [])
                if str(feature or "").strip()
            ]
            if not relative_path:
                continue
            target = self.workspace.resolve_path(working_dir / relative_path)
            if not target.exists():
                failures.append(f"Missing HTML file: {relative_path}")
                continue

            parser = _HTMLArtifactParser()
            parser.feed(target.read_text(encoding="utf-8"))
            checked += 1

            missing_refs: list[str] = []
            script_texts: list[tuple[str, str]] = []
            css_texts: list[tuple[str, str]] = []
            for reference in parser.references:
                resolved = (target.parent / reference).resolve()
                if not resolved.exists():
                    missing_refs.append(reference)
                    continue
                if Path(reference).suffix.lower() in {".js", ".mjs", ".cjs"}:
                    script_texts.append((reference, resolved.read_text(encoding="utf-8")))
                if Path(reference).suffix.lower() == ".css":
                    css_texts.append((reference, resolved.read_text(encoding="utf-8")))

            if missing_refs:
                failures.extend(f"{relative_path} -> {reference}" for reference in missing_refs)
                continue

            script_texts.extend(
                (f"{relative_path}#inline-script-{index + 1}", content)
                for index, content in enumerate(parser.inline_scripts)
            )
            syntax_failures = self._javascript_syntax_failures(script_texts, node_binary=node_binary)
            failures.extend(f"{relative_path}: {message}" for message in syntax_failures)

            html_token_space = parser.token_space
            script_token_space = self._normalized_web_token_space(content for _, content in script_texts)
            css_token_space = self._normalized_web_token_space(content for _, content in css_texts)
            token_space = self._normalized_web_token_space(
                [html_token_space, script_token_space, css_token_space]
            )
            missing_dom_ids = self._missing_dom_ids(
                html_ids=parser.ids,
                script_texts=script_texts,
            )
            if missing_dom_ids:
                failures.append(
                    f"{relative_path}: missing DOM ids referenced by JS ({', '.join(missing_dom_ids)})"
                )
            missing_features = [
                feature
                for feature in expected_features
                if not self._web_feature_present(
                    feature,
                    token_space,
                    html_token_space=html_token_space,
                    script_token_space=script_token_space,
                    css_token_space=css_token_space,
                )
            ]
            if missing_features:
                failures.append(
                    f"{relative_path}: missing expected web features ({', '.join(missing_features)})"
                )

            dom_markers = parser.structural_markers
            js_summary = (
                f"JS parsed: {len(script_texts)} source(s)"
                if node_binary
                else f"JS parse skipped: node unavailable ({len(script_texts)} source(s))"
            )
            feature_summary = ", ".join(expected_features) if expected_features else "none inferred"
            marker_summary = ", ".join(dom_markers[:5]) if dom_markers else "no obvious interactive markers"
            summaries.append(
                f"{relative_path}: refs ok; {js_summary}; expected features: {feature_summary}; markers: {marker_summary}."
            )

        success = not failures and checked > 0
        notes = "Structural web checks only; no browser/runtime smoke test was executed."
        stdout_parts = [part for part in summaries if part]
        stdout_parts.append(notes)
        return {
            "success": success,
            "message": (
                "Structural web validation passed."
                if success
                else "Structural web validation failed."
            ),
            "risk_level": "low",
            "stdout": "\n".join(stdout_parts).strip(),
            "stderr": "\n".join(failures),
            "exit_code": 0 if success else 1,
            "command": command,
        }

    def _run_web_runtime_smoke_validation(self, command: str, working_dir: Path, payload_items: list) -> dict:
        node_binary = shutil.which("node")
        if not node_binary:
            return {
                "success": False,
                "message": "Web runtime smoke test could not start because Node is unavailable.",
                "risk_level": "low",
                "stdout": "",
                "stderr": "Node is required for internal:web_runtime_smoke validation.",
                "exit_code": 1,
                "command": command,
            }

        failures: list[str] = []
        summaries: list[str] = []
        checked = 0
        max_targets = 2

        for item in payload_items[:max_targets]:
            descriptor = item if isinstance(item, dict) else {"path": item}
            relative_path = str(descriptor.get("path") or "").strip()
            if not relative_path:
                continue
            target = self.workspace.resolve_path(working_dir / relative_path)
            if not target.exists():
                failures.append(f"Missing HTML file: {relative_path}")
                continue

            parser = _HTMLArtifactParser()
            parser.feed(target.read_text(encoding="utf-8"))
            checked += 1

            missing_refs: list[str] = []
            script_texts: list[tuple[str, str]] = []
            for reference in parser.references:
                resolved = (target.parent / reference).resolve()
                if not resolved.exists():
                    missing_refs.append(reference)
                    continue
                if Path(reference).suffix.lower() in {".js", ".mjs", ".cjs"}:
                    script_texts.append((reference, resolved.read_text(encoding="utf-8")))

            if missing_refs:
                failures.extend(f"{relative_path} -> {reference}" for reference in missing_refs)
                continue

            script_texts.extend(
                (f"{relative_path}#inline-script-{index + 1}", content)
                for index, content in enumerate(parser.inline_scripts)
            )

            runtime_result = self._execute_web_runtime_smoke(
                node_binary=node_binary,
                relative_path=relative_path,
                elements=parser.elements,
                script_texts=script_texts,
            )
            summaries.append(str(runtime_result.get("stdout") or "").strip())
            if not runtime_result["success"]:
                failure_text = str(runtime_result.get("stderr") or runtime_result.get("stdout") or "").strip()
                failures.append(f"{relative_path}: {failure_text or 'Runtime smoke failed.'}")

        success = not failures and checked > 0
        return {
            "success": success,
            "message": (
                "Runtime web smoke validation passed."
                if success
                else "Runtime web smoke validation failed."
            ),
            "risk_level": "low",
            "stdout": "\n".join(part for part in summaries if part).strip(),
            "stderr": "\n".join(failures),
            "exit_code": 0 if success else 1,
            "command": command,
        }

    def _execute_web_runtime_smoke(
        self,
        *,
        node_binary: str,
        relative_path: str,
        elements: list[dict[str, object]],
        script_texts: list[tuple[str, str]],
    ) -> dict[str, object]:
        payload = {
            "path": relative_path,
            "elements": elements,
            "scripts": [
                {"label": label, "content": content}
                for label, content in script_texts
            ],
        }
        harness = textwrap.dedent(
            """
            const fs = require("fs");
            const vm = require("vm");

            const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

            class EventTarget {
              constructor() {
                this._listeners = new Map();
              }

              addEventListener(type, handler) {
                if (!type || typeof handler !== "function") {
                  return;
                }
                const bucket = this._listeners.get(type) || [];
                bucket.push(handler);
                this._listeners.set(type, bucket);
              }

              removeEventListener(type, handler) {
                const bucket = this._listeners.get(type);
                if (!bucket) {
                  return;
                }
                this._listeners.set(
                  type,
                  bucket.filter((candidate) => candidate !== handler),
                );
              }

              dispatchEvent(event) {
                if (!event || !event.type) {
                  throw new Error("Event object with a type is required.");
                }
                event.target = event.target || this;
                event.currentTarget = this;
                const bucket = this._listeners.get(event.type) || [];
                for (const handler of bucket) {
                  handler.call(this, event);
                }
                return !event.defaultPrevented;
              }
            }

            class BasicEvent {
              constructor(type, init = {}) {
                this.type = type;
                this.bubbles = Boolean(init.bubbles);
                this.cancelable = Boolean(init.cancelable);
                this.defaultPrevented = false;
                this.detail = init.detail;
                this.key = init.key || "";
                this.target = init.target || null;
                this.currentTarget = null;
              }

              preventDefault() {
                if (this.cancelable) {
                  this.defaultPrevented = true;
                }
              }
            }

            class BasicCustomEvent extends BasicEvent {
              constructor(type, init = {}) {
                super(type, init);
                this.detail = init.detail;
              }
            }

            class ClassList {
              constructor(owner, values = []) {
                this.owner = owner;
                this.values = new Set(values.filter(Boolean));
              }

              add(...tokens) {
                for (const token of tokens) {
                  if (token) {
                    this.values.add(token);
                  }
                }
                this.owner._syncClassAttr();
              }

              remove(...tokens) {
                for (const token of tokens) {
                  this.values.delete(token);
                }
                this.owner._syncClassAttr();
              }

              contains(token) {
                return this.values.has(token);
              }

              toggle(token, force) {
                if (force === true) {
                  this.values.add(token);
                  this.owner._syncClassAttr();
                  return true;
                }
                if (force === false) {
                  this.values.delete(token);
                  this.owner._syncClassAttr();
                  return false;
                }
                if (this.values.has(token)) {
                  this.values.delete(token);
                  this.owner._syncClassAttr();
                  return false;
                }
                this.values.add(token);
                this.owner._syncClassAttr();
                return true;
              }

              toString() {
                return [...this.values].join(" ");
              }
            }

            class Element extends EventTarget {
              constructor(descriptor = {}) {
                super();
                this.tagName = String(descriptor.tag || "div").toUpperCase();
                this.nodeName = this.tagName;
                this.id = String(descriptor.id || "");
                this.name = String(descriptor.name || "");
                this.type = String(descriptor.type || "");
                this.value = "";
                this.checked = false;
                this.textContent = "";
                this.innerHTML = "";
                this.children = [];
                this.parentNode = null;
                this.ownerDocument = null;
                this.style = {};
                this.dataset = {};
                this.attributes = {};
                this.classList = new ClassList(this, descriptor.classes || []);
                this._applyDescriptor(descriptor);
              }

              _applyDescriptor(descriptor) {
                const attrs = descriptor.attributes || {};
                for (const [name, value] of Object.entries(attrs)) {
                  this.setAttribute(name, value);
                }
                if (descriptor.id) {
                  this.setAttribute("id", descriptor.id);
                }
                if (descriptor.name) {
                  this.setAttribute("name", descriptor.name);
                }
                if (descriptor.type) {
                  this.setAttribute("type", descriptor.type);
                }
                for (const klass of descriptor.classes || []) {
                  this.classList.add(klass);
                }
                this.value = this.tagName === "SELECT" ? "dark" : "demo";
                if (this.type === "checkbox") {
                  this.checked = true;
                  this.value = "dark";
                }
              }

              _syncClassAttr() {
                const value = this.classList.toString();
                if (value) {
                  this.attributes.class = value;
                } else {
                  delete this.attributes.class;
                }
              }

              setAttribute(name, value) {
                const normalizedName = String(name || "");
                const normalizedValue = String(value ?? "");
                this.attributes[normalizedName] = normalizedValue;
                if (normalizedName === "id") {
                  this.id = normalizedValue;
                } else if (normalizedName === "name") {
                  this.name = normalizedValue;
                } else if (normalizedName === "type") {
                  this.type = normalizedValue;
                } else if (normalizedName === "class") {
                  this.classList = new ClassList(this, normalizedValue.split(/\\s+/).filter(Boolean));
                } else if (normalizedName.startsWith("data-")) {
                  const key = normalizedName
                    .slice(5)
                    .replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
                  this.dataset[key] = normalizedValue;
                }
              }

              getAttribute(name) {
                return Object.prototype.hasOwnProperty.call(this.attributes, name)
                  ? this.attributes[name]
                  : null;
              }

              removeAttribute(name) {
                delete this.attributes[name];
                if (name === "id") {
                  this.id = "";
                }
                if (name === "class") {
                  this.classList = new ClassList(this, []);
                }
              }

              appendChild(child) {
                child.parentNode = this;
                this.children.push(child);
                return child;
              }

              querySelector(selector) {
                return this.ownerDocument ? this.ownerDocument.querySelector(selector) : null;
              }

              querySelectorAll(selector) {
                return this.ownerDocument ? this.ownerDocument.querySelectorAll(selector) : [];
              }

              focus() {}

              blur() {}

              reset() {
                if (!this.ownerDocument) {
                  return;
                }
                for (const element of this.ownerDocument.elements) {
                  if (element === this) {
                    continue;
                  }
                  if (["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName)) {
                    element.value = "";
                    if (element.type === "checkbox") {
                      element.checked = false;
                    }
                  }
                }
              }
            }

            function buildStorage() {
              const values = new Map();
              return {
                getItem(key) {
                  return values.has(key) ? values.get(key) : null;
                },
                setItem(key, value) {
                  values.set(String(key), String(value));
                },
                removeItem(key) {
                  values.delete(String(key));
                },
                clear() {
                  values.clear();
                },
              };
            }

            class Document extends EventTarget {
              constructor(descriptors = []) {
                super();
                this.readyState = "complete";
                this.elements = [];
                this.documentElement = new Element({ tag: "html" });
                this.body = new Element({ tag: "body" });
                this.documentElement.ownerDocument = this;
                this.body.ownerDocument = this;
                this.documentElement.appendChild(this.body);
                this.elements.push(this.documentElement, this.body);

                for (const descriptor of descriptors) {
                  const element = new Element(descriptor);
                  element.ownerDocument = this;
                  this.body.appendChild(element);
                  this.elements.push(element);
                }
              }

              createElement(tagName) {
                const element = new Element({ tag: tagName });
                element.ownerDocument = this;
                return element;
              }

              getElementById(id) {
                return this.elements.find((element) => element.id === String(id)) || null;
              }

              querySelector(selector) {
                return this.querySelectorAll(selector)[0] || null;
              }

              querySelectorAll(selector) {
                const normalized = String(selector || "").trim();
                if (!normalized) {
                  return [];
                }
                if (normalized === "body") {
                  return [this.body];
                }
                if (normalized === "html") {
                  return [this.documentElement];
                }
                if (normalized.startsWith("#")) {
                  const match = this.getElementById(normalized.slice(1));
                  return match ? [match] : [];
                }
                if (normalized.startsWith(".")) {
                  const token = normalized.slice(1);
                  return this.elements.filter((element) => element.classList.contains(token));
                }

                const attrMatch = normalized.match(/^([a-zA-Z0-9_-]+)?\\[(.+?)\\]$/);
                if (attrMatch) {
                  const tagName = attrMatch[1] ? attrMatch[1].toUpperCase() : null;
                  const [rawAttr, rawValue] = attrMatch[2].split("=");
                  const attrName = String(rawAttr || "").trim();
                  const attrValue = rawValue
                    ? String(rawValue).trim().replace(/^["']|["']$/g, "")
                    : null;
                  return this.elements.filter((element) => {
                    if (tagName && element.tagName !== tagName) {
                      return false;
                    }
                    const current = element.getAttribute(attrName);
                    if (current === null) {
                      return false;
                    }
                    if (attrValue === null) {
                      return true;
                    }
                    return current === attrValue;
                  });
                }

                return this.elements.filter((element) => element.tagName === normalized.toUpperCase());
              }
            }

            async function main() {
              const document = new Document(payload.elements || []);
              const localStorage = buildStorage();
              const sessionStorage = buildStorage();
              const windowTarget = new EventTarget();
              const windowObject = {
                window: null,
                self: null,
                globalThis: null,
                document,
                navigator: { userAgent: "marc-web-runtime-smoke" },
                location: {
                  href: "http://localhost/",
                  reload() {},
                },
                localStorage,
                sessionStorage,
                console,
                setTimeout,
                clearTimeout,
                setInterval,
                clearInterval,
                requestAnimationFrame(callback) {
                  return setTimeout(() => callback(Date.now()), 0);
                },
                cancelAnimationFrame(handle) {
                  clearTimeout(handle);
                },
                matchMedia() {
                  return {
                    matches: false,
                    addEventListener() {},
                    removeEventListener() {},
                    addListener() {},
                    removeListener() {},
                  };
                },
                getComputedStyle() {
                  return {
                    getPropertyValue() {
                      return "";
                    },
                  };
                },
                alert() {},
                confirm() {
                  return true;
                },
                Event: BasicEvent,
                CustomEvent: BasicCustomEvent,
                Element,
                HTMLElement: Element,
                Node: Element,
                addEventListener: (...args) => windowTarget.addEventListener(...args),
                removeEventListener: (...args) => windowTarget.removeEventListener(...args),
                dispatchEvent: (...args) => windowTarget.dispatchEvent(...args),
              };
              windowObject.window = windowObject;
              windowObject.self = windowObject;
              windowObject.globalThis = windowObject;
              document.defaultView = windowObject;
              document.location = windowObject.location;

              const context = vm.createContext(windowObject);
              context.document = document;
              context.window = windowObject;
              context.self = windowObject;
              context.globalThis = windowObject;
              context.localStorage = localStorage;
              context.sessionStorage = sessionStorage;
              context.navigator = windowObject.navigator;
              context.location = windowObject.location;
              context.Event = BasicEvent;
              context.CustomEvent = BasicCustomEvent;
              context.Element = Element;
              context.HTMLElement = Element;
              context.Node = Element;
              context.console = console;
              context.setTimeout = setTimeout;
              context.clearTimeout = clearTimeout;
              context.setInterval = setInterval;
              context.clearInterval = clearInterval;
              context.requestAnimationFrame = windowObject.requestAnimationFrame;
              context.cancelAnimationFrame = windowObject.cancelAnimationFrame;
              context.matchMedia = windowObject.matchMedia;
              context.getComputedStyle = windowObject.getComputedStyle;
              context.alert = windowObject.alert;
              context.confirm = windowObject.confirm;
              context.global = windowObject;

              const failures = [];
              process.on("uncaughtException", (error) => {
                failures.push(error);
              });

              for (const script of payload.scripts || []) {
                const body = String(script.content || "");
                if (!body.trim()) {
                  continue;
                }
                const source = new vm.Script(body, { filename: script.label || payload.path || "inline-script.js" });
                source.runInContext(context, { timeout: 2000 });
              }

              document.dispatchEvent(new BasicEvent("DOMContentLoaded", { bubbles: true }));
              windowObject.dispatchEvent(new BasicEvent("load"));

              for (const element of document.elements) {
                if (["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName)) {
                  if (!element.value) {
                    element.value = element.type === "checkbox" ? "dark" : "demo";
                  }
                  element.dispatchEvent(new BasicEvent("input", { bubbles: true, target: element }));
                  element.dispatchEvent(new BasicEvent("change", { bubbles: true, target: element }));
                }
              }
              for (const element of document.elements) {
                if (element.tagName === "BUTTON") {
                  element.dispatchEvent(new BasicEvent("click", { bubbles: true, target: element }));
                }
              }
              for (const element of document.elements) {
                if (element.tagName === "FORM") {
                  element.dispatchEvent(new BasicEvent("submit", { bubbles: true, cancelable: true, target: element }));
                }
              }

              await new Promise((resolve) => setTimeout(resolve, 15));

              if (failures.length > 0) {
                throw failures[0];
              }

              const scriptCount = Array.isArray(payload.scripts) ? payload.scripts.length : 0;
              console.log(`${payload.path}: runtime smoke passed; scripts executed: ${scriptCount}; synthetic events: DOMContentLoaded, load, input, change, click, submit.`);
            }

            main().catch((error) => {
              const detail = error && error.stack ? error.stack : String(error);
              console.error(detail);
              process.exit(1);
            });
            """
        ).strip()
        payload_text = json.dumps(payload, ensure_ascii=False)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as payload_file:
            payload_file.write(payload_text)
            payload_path = payload_file.name
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".cjs", delete=False) as harness_file:
            harness_file.write(harness)
            harness_path = harness_file.name
        try:
            completed = subprocess.run(
                [node_binary, harness_path, payload_path],
                cwd=self.workspace.root,
                text=True,
                capture_output=True,
                timeout=min(max(int(self.config.shell_timeout), 1), 12),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = str(exc.stdout or "")[-self.config.max_read_chars :]
            stderr = str(exc.stderr or "")[-self.config.max_read_chars :]
            return {
                "success": False,
                "stdout": stdout,
                "stderr": stderr or "Runtime smoke timed out.",
            }
        finally:
            Path(payload_path).unlink(missing_ok=True)
            Path(harness_path).unlink(missing_ok=True)

        return {
            "success": completed.returncode == 0,
            "stdout": str(completed.stdout or "")[-self.config.max_read_chars :].strip(),
            "stderr": str(completed.stderr or "")[-self.config.max_read_chars :].strip(),
        }

    def _normalized_web_token_space(self, parts) -> str:
        if not isinstance(parts, (list, tuple)):
            parts = list(parts)
        normalized = " ".join(str(part or "") for part in parts).lower()
        normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
        return f" {normalized} "

    def _default_python_smoke_input(self, target: Path) -> str:
        try:
            content = target.read_text(encoding="utf-8")
        except OSError:
            return "\n"

        input_calls = content.count("input(")
        if input_calls <= 0:
            return ""

        seed_values = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "n",
            "q",
            "quit",
            "exit",
            "0",
            "Alice",
            "Bob",
        ]
        line_count = min(max(input_calls + 4, 9), len(seed_values))
        return "\n".join(seed_values[:line_count]) + "\n"

    def _javascript_syntax_failures(
        self,
        script_texts: list[tuple[str, str]],
        *,
        node_binary: str | None,
    ) -> list[str]:
        if not script_texts or not node_binary:
            return []

        failures: list[str] = []
        for label, content in script_texts:
            temp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    suffix=".js",
                    delete=False,
                ) as handle:
                    handle.write(content)
                    temp_path = handle.name
                completed = subprocess.run(
                    [node_binary, "--check", temp_path],
                    text=True,
                    capture_output=True,
                    timeout=min(max(int(self.config.shell_timeout), 1), 8),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{label}: JavaScript syntax check timed out.")
                continue
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)

            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip() or "JavaScript parse failed."
                failures.append(f"{label}: {message}")
        return failures

    def _web_feature_present(
        self,
        feature: str,
        token_space: str,
        *,
        html_token_space: str = "",
        script_token_space: str = "",
        css_token_space: str = "",
    ) -> bool:
        normalized = str(feature or "").strip().lower()
        if not normalized:
            return True
        keyword_groups = {
            "menu": (" menu ", " nav ", " navigation ", " menue ", " menü "),
            "highscore": (
                " highscore ",
                " high score ",
                " scoreboard ",
                " leaderboard ",
                " best score ",
                " bestenliste ",
            ),
            "score": (" score ", " punkte ", " punktestand ", " scoreboard "),
            "start_controls": (
                " start ",
                " play ",
                " pause ",
                " resume ",
                " restart ",
                " reset ",
                " reload ",
                " neustart ",
                " fortsetzen ",
            ),
            "keyboard_controls": (
                " keyboard ",
                " tastatur ",
                " keydown ",
                " keyup ",
                " arrow ",
                " arrowup ",
                " arrowdown ",
                " arrowleft ",
                " arrowright ",
                " pfeiltaste ",
                " pfeiltasten ",
            ),
            "game_over": (
                " game over ",
                " gameover ",
                " alert ",
                " neustart ",
                " restart ",
                " reset ",
                " reload ",
            ),
            "dialog": (" dialog ", " modal ", " popup ", " overlay "),
            "canvas": (" canvas ", " spielfeld ", " game board "),
            "settings": (" settings ", " options ", " config ", " einstellungen "),
        }
        markers = keyword_groups.get(normalized)
        if markers is None:
            return f" {normalized} " in token_space
        return any(marker in token_space for marker in markers)

    def _missing_dom_ids(
        self,
        *,
        html_ids: list[str],
        script_texts: list[tuple[str, str]],
    ) -> list[str]:
        known_ids = {
            str(value or "").strip()
            for value in html_ids
            if str(value or "").strip()
        }
        if not known_ids or not script_texts:
            return []

        referenced: set[str] = set()
        patterns = [
            re.compile(r"""getElementById\(\s*["']([^"']+)["']\s*\)"""),
            re.compile(r"""querySelector\(\s*["']#([^"']+)["']\s*\)"""),
        ]
        for _, content in script_texts:
            source = str(content or "")
            for pattern in patterns:
                referenced.update(
                    match.group(1).strip()
                    for match in pattern.finditer(source)
                    if match.group(1).strip()
                )
        return sorted(identifier for identifier in referenced if identifier not in known_ids)


class _HTMLReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def reset_refs(self) -> None:
        self.references = []
        self.reset()

    def handle_starttag(self, tag: str, attrs) -> None:
        del tag
        for key, value in attrs:
            if key not in {"src", "href"} or not value:
                continue
            cleaned = str(value).split("#", 1)[0].split("?", 1)[0].strip()
            if not cleaned or cleaned.startswith(("http://", "https://", "data:", "mailto:", "javascript:", "#")):
                continue
            self.references.append(cleaned)


class _HTMLArtifactParser(_HTMLReferenceParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[dict[str, object]] = []
        self.inline_scripts: list[str] = []
        self.visible_text: list[str] = []
        self.ids: list[str] = []
        self.classes: list[str] = []
        self.tags: list[str] = []
        self.structural_markers: list[str] = []
        self._script_buffer: list[str] = []
        self._inside_script = False

    def reset_refs(self) -> None:
        super().reset_refs()
        self.elements = []
        self.inline_scripts = []
        self.visible_text = []
        self.ids = []
        self.classes = []
        self.tags = []
        self.structural_markers = []
        self._script_buffer = []
        self._inside_script = False

    @property
    def token_space(self) -> str:
        parts = self.tags + self.ids + self.classes + self.visible_text + self.inline_scripts
        normalized = " ".join(parts).lower()
        normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
        return f" {normalized} "

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = {str(key): str(value) for key, value in attrs if value}
        descriptor: dict[str, object] = {"tag": tag}
        if attrs_dict.get("id"):
            descriptor["id"] = attrs_dict["id"]
        if attrs_dict.get("class"):
            descriptor["classes"] = attrs_dict["class"].split()
        if attrs_dict.get("name"):
            descriptor["name"] = attrs_dict["name"]
        if attrs_dict.get("type"):
            descriptor["type"] = attrs_dict["type"]
        if attrs_dict:
            descriptor["attributes"] = attrs_dict
        self.elements.append(descriptor)
        self.tags.append(tag)
        if attrs_dict.get("id"):
            self.ids.append(attrs_dict["id"])
        if attrs_dict.get("class"):
            self.classes.extend(attrs_dict["class"].split())

        interactive_tags = {"button", "canvas", "dialog", "form", "input", "menu", "select"}
        if tag in interactive_tags and tag not in self.structural_markers:
            self.structural_markers.append(tag)
        if any(key.startswith("on") for key in attrs_dict):
            if "dom-events" not in self.structural_markers:
                self.structural_markers.append("dom-events")

        if tag == "script" and not attrs_dict.get("src"):
            self._inside_script = True
            self._script_buffer = []
        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._inside_script:
            script_text = "".join(self._script_buffer).strip()
            if script_text:
                self.inline_scripts.append(script_text)
                if "inline-script" not in self.structural_markers:
                    self.structural_markers.append("inline-script")
            self._inside_script = False
            self._script_buffer = []

    def handle_data(self, data: str) -> None:
        text = str(data or "")
        if self._inside_script:
            self._script_buffer.append(text)
            return
        cleaned = " ".join(text.split()).strip()
        if cleaned:
            self.visible_text.append(cleaned)
