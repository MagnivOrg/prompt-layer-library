from __future__ import annotations

import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from promptlayer.cli.setup.agents import SetupAgent, skill_package_path
from promptlayer.cli.setup.constants import (
    PYTHON_SDK_SKILL_APPENDIX,
    SDK_EVALS_SKILLS_ZIP_URL,
    SKILL_NAME,
    SKILL_SOURCE_URLS,
)

FetchText = Callable[[str], str]
FetchBinary = Callable[[str], bytes]
ZipEntries = Dict[str, bytes]


@dataclass
class SkillInstallResult:
    agent: SetupAgent
    skill_name: str
    path: str
    status: str  # written | skipped | updated


def default_fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "promptlayer-cli"})
    with urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8")


def default_fetch_binary(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "promptlayer-cli"})
    with urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def fetch_promptlayer_skill(fetch_text: FetchText = default_fetch_text) -> str:
    errors: List[str] = []
    for url in SKILL_SOURCE_URLS:
        try:
            content = fetch_text(url).rstrip()
            if not content:
                errors.append(f"{url}: empty response")
                continue
            return ensure_python_sdk_appendix(content)
        except (HTTPError, URLError, OSError, UnicodeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download the PromptLayer skill.\n" + "\n".join(errors))


def fetch_sdk_evals_skill_entries(
    fetch_binary: FetchBinary = default_fetch_binary,
    *,
    url: str = SDK_EVALS_SKILLS_ZIP_URL,
) -> ZipEntries:
    return read_zip_entries(fetch_binary(url))


def ensure_python_sdk_appendix(skill_content: str) -> str:
    trimmed = skill_content.rstrip()
    marker = "Python SDK guidance (from `promptlayer setup`)"
    if marker in trimmed:
        return f"{trimmed}\n"
    return f"{trimmed}{PYTHON_SDK_SKILL_APPENDIX}"


def install_skill_for_agents(
    *,
    cwd: Path,
    agents: Sequence[SetupAgent],
    skill_content: str,
    skill_name: str = SKILL_NAME,
    force: bool = False,
) -> List[SkillInstallResult]:
    files: ZipEntries = {f"{skill_name}/SKILL.md": skill_content.encode("utf-8")}
    return install_skill_entries_for_agents(cwd=cwd, agents=agents, files=files, force=force)


def install_skill_entries_for_agents(
    *,
    cwd: Path,
    agents: Sequence[SetupAgent],
    files: Mapping[str, bytes],
    force: bool = False,
) -> List[SkillInstallResult]:
    skill_names = sorted(
        {
            entry.split("/", 1)[0]
            for entry in files
            if entry and "/" in entry and not entry.endswith("/")
        }
    )
    results: List[SkillInstallResult] = []
    handled: set[str] = set()

    for agent in agents:
        for skill_name in skill_names:
            relative_root = skill_package_path(agent, skill_name)
            dedupe_key = f"{agent.skills_dir}:{skill_name}"
            if dedupe_key in handled:
                results.append(
                    SkillInstallResult(
                        agent=agent,
                        skill_name=skill_name,
                        path=relative_root,
                        status="skipped",
                    )
                )
                continue
            handled.add(dedupe_key)

            absolute_root = (cwd / relative_root).resolve()
            root_exists = absolute_root.exists()
            if root_exists and not force:
                results.append(
                    SkillInstallResult(
                        agent=agent,
                        skill_name=skill_name,
                        path=relative_root,
                        status="skipped",
                    )
                )
                continue

            status = "updated" if root_exists else "written"
            for entry_path, content in files.items():
                if not entry_path.startswith(f"{skill_name}/"):
                    continue
                absolute_path = (cwd / agent.skills_dir / entry_path).resolve()
                skills_root = (cwd / agent.skills_dir).resolve()
                if skills_root not in absolute_path.parents and absolute_path != skills_root:
                    raise RuntimeError(f"Refusing unsafe skill path: {entry_path}")
                absolute_path.parent.mkdir(parents=True, exist_ok=True)
                absolute_path.write_bytes(content)

            results.append(
                SkillInstallResult(
                    agent=agent,
                    skill_name=skill_name,
                    path=relative_root,
                    status=status,
                )
            )
    return results


def describe_skill_install(result: SkillInstallResult) -> str:
    label = f"{result.agent.label} skill ({result.skill_name})"
    if result.status == "skipped":
        return f"Skipped {label}: {result.path} already exists (use --force to overwrite)"
    if result.status == "updated":
        return f"Updated {label}: {result.path}"
    return f"Installed {label}: {result.path}"


def read_zip_entries(data: bytes) -> ZipEntries:
    entries: ZipEntries = {}
    with zipfile.ZipFile(BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Refusing unsafe zip member path: {info.filename}")
            entries[info.filename.replace("\\", "/")] = archive.read(info)
    return entries
