from typing import Dict, List, Literal, Optional, TypedDict, Union

from typing_extensions import Required

SkillProvider = Literal["claude_code", "openai", "openclaw"]


class InitialSkillFileUpdate(TypedDict, total=False):
    path: Required[str]
    content: str


class SkillFileUpdate(TypedDict, total=False):
    path: Required[str]
    content: str


class SkillFileMove(TypedDict):
    old_path: Required[str]
    new_path: Required[str]


class CreateSkillCollection(TypedDict, total=False):
    name: Required[str]
    folder_id: int
    provider: Optional[SkillProvider]
    files: List[InitialSkillFileUpdate]
    commit_message: Optional[str]


class SaveSkillCollectionVersion(TypedDict, total=False):
    file_updates: List[SkillFileUpdate]
    moves: List[SkillFileMove]
    deletes: List[str]
    commit_message: Optional[str]
    release_label: Optional[str]
    provider: Optional[SkillProvider]


class UpdateSkillCollection(SaveSkillCollectionVersion, total=False):
    name: Optional[str]


class SkillCollection(TypedDict, total=False):
    id: Required[str]
    workspace_id: Required[int]
    folder_id: Optional[int]
    root_path: Required[str]
    name: Required[str]
    provider: Optional[SkillProvider]
    is_deleted: Required[bool]
    created_by: Optional[int]
    created_at: Required[str]
    updated_by: Optional[int]
    updated_at: Required[str]


class SkillCollectionVersion(TypedDict, total=False):
    id: Required[str]
    skill_collection_id: Required[str]
    workspace_id: Required[int]
    number: Required[int]
    root_path_at_version: Required[str]
    file_paths: List[str]
    commit_message: Optional[str]
    release_label: Optional[str]
    archived: Required[bool]
    created_by: Optional[int]
    user_email: Optional[str]
    created_at: Required[str]


class CreateSkillCollectionResponse(TypedDict):
    success: Required[bool]
    skill_collection: Required[SkillCollection]


class PullSkillCollectionResponse(TypedDict):
    success: Required[bool]
    skill_collection: Required[SkillCollection]
    files: Required[Dict[str, str]]
    version: Union[SkillCollectionVersion, None]


class UpdateSkillCollectionResponse(TypedDict, total=False):
    success: Required[bool]
    skill_collection: Required[SkillCollection]
    version: Optional[SkillCollectionVersion]
