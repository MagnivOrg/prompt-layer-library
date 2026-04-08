from typing import Any, List, Optional, Union

from promptlayer import exceptions as _exceptions
from promptlayer.skills.api import (
    acreate_skill_collection,
    apull_skill_collection,
    asave_skill_collection_version,
    aupdate_skill_collection,
    create_skill_collection,
    pull_skill_collection,
    save_skill_collection_version,
    update_skill_collection,
)
from promptlayer.types.skill import (
    CreateSkillCollection,
    CreateSkillCollectionResponse,
    PullSkillCollectionResponse,
    SaveSkillCollectionVersion,
    SkillFileMove,
    SkillFileUpdate,
    UpdateSkillCollectionResponse,
)


def _has_version_updates(
    *,
    file_updates: Optional[List[SkillFileUpdate]],
    moves: Optional[List[SkillFileMove]],
    deletes: Optional[List[str]],
    commit_message: Optional[str],
    release_label: Optional[str],
    provider: Optional[str],
    zip: Any = None,
) -> bool:
    return any(
        value is not None for value in (file_updates, moves, deletes, commit_message, release_label, provider, zip)
    )


def _build_version_payload(
    *,
    file_updates: Optional[List[SkillFileUpdate]],
    moves: Optional[List[SkillFileMove]],
    deletes: Optional[List[str]],
    commit_message: Optional[str],
    release_label: Optional[str],
    provider: Optional[str],
) -> SaveSkillCollectionVersion:
    payload: SaveSkillCollectionVersion = {}
    if file_updates is not None:
        payload["file_updates"] = file_updates
    if moves is not None:
        payload["moves"] = moves
    if deletes is not None:
        payload["deletes"] = deletes
    if commit_message is not None:
        payload["commit_message"] = commit_message
    if release_label is not None:
        payload["release_label"] = release_label
    if provider is not None:
        payload["provider"] = provider
    return payload


class SkillManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    def pull(
        self,
        identifier: str,
        *,
        label: Union[str, None] = None,
        version: Union[int, None] = None,
        format: Union[str, None] = None,
    ) -> Union[PullSkillCollectionResponse, bytes, None]:
        return pull_skill_collection(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            identifier,
            label=label,
            version=version,
            format=format,
        )

    def create(self, body: CreateSkillCollection, zip: Any = None) -> Union[CreateSkillCollectionResponse, None]:
        if zip is None:
            return create_skill_collection(self.api_key, self.base_url, self.throw_on_error, body)
        return create_skill_collection(self.api_key, self.base_url, self.throw_on_error, body, zip=zip)

    def update(
        self,
        identifier: str,
        *,
        name: Union[str, None] = None,
        file_updates: Optional[List[SkillFileUpdate]] = None,
        moves: Optional[List[SkillFileMove]] = None,
        deletes: Optional[List[str]] = None,
        commit_message: Union[str, None] = None,
        release_label: Union[str, None] = None,
        provider: Union[str, None] = None,
        zip: Any = None,
    ) -> Union[UpdateSkillCollectionResponse, None]:
        rename_requested = name is not None
        version_updates_requested = _has_version_updates(
            file_updates=file_updates,
            moves=moves,
            deletes=deletes,
            commit_message=commit_message,
            release_label=release_label,
            provider=provider,
            zip=zip,
        )

        if not rename_requested and not version_updates_requested:
            raise _exceptions.PromptLayerValidationError(
                "Please provide at least one skill collection update.", response=None, body=None
            )

        skill_collection = None
        effective_identifier = identifier

        if rename_requested:
            rename_response = update_skill_collection(
                self.api_key,
                self.base_url,
                self.throw_on_error,
                identifier,
                {"name": name},
            )
            if not rename_response:
                return None
            skill_collection = rename_response["skill_collection"]
            effective_identifier = skill_collection["id"]

        version_response = None
        if version_updates_requested:
            save_payload = _build_version_payload(
                file_updates=file_updates,
                moves=moves,
                deletes=deletes,
                commit_message=commit_message,
                release_label=release_label,
                provider=provider,
            )
            if zip is None:
                version_response = save_skill_collection_version(
                    self.api_key,
                    self.base_url,
                    self.throw_on_error,
                    effective_identifier,
                    save_payload,
                )
            else:
                version_response = save_skill_collection_version(
                    self.api_key,
                    self.base_url,
                    self.throw_on_error,
                    effective_identifier,
                    save_payload,
                    zip=zip,
                )
            if not version_response:
                return None

        if skill_collection is None:
            pull_response = self.pull(effective_identifier)
            if not pull_response or isinstance(pull_response, bytes):
                return None
            skill_collection = pull_response["skill_collection"]

        response: UpdateSkillCollectionResponse = {
            "success": True,
            "skill_collection": skill_collection,
        }
        if version_response:
            response["version"] = version_response["version"]
        return response


class AsyncSkillManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error

    async def pull(
        self,
        identifier: str,
        *,
        label: Union[str, None] = None,
        version: Union[int, None] = None,
        format: Union[str, None] = None,
    ) -> Union[PullSkillCollectionResponse, bytes, None]:
        return await apull_skill_collection(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            identifier,
            label=label,
            version=version,
            format=format,
        )

    async def create(self, body: CreateSkillCollection, zip: Any = None) -> Union[CreateSkillCollectionResponse, None]:
        if zip is None:
            return await acreate_skill_collection(self.api_key, self.base_url, self.throw_on_error, body)
        return await acreate_skill_collection(self.api_key, self.base_url, self.throw_on_error, body, zip=zip)

    async def update(
        self,
        identifier: str,
        *,
        name: Union[str, None] = None,
        file_updates: Optional[List[SkillFileUpdate]] = None,
        moves: Optional[List[SkillFileMove]] = None,
        deletes: Optional[List[str]] = None,
        commit_message: Union[str, None] = None,
        release_label: Union[str, None] = None,
        provider: Union[str, None] = None,
        zip: Any = None,
    ) -> Union[UpdateSkillCollectionResponse, None]:
        rename_requested = name is not None
        version_updates_requested = _has_version_updates(
            file_updates=file_updates,
            moves=moves,
            deletes=deletes,
            commit_message=commit_message,
            release_label=release_label,
            provider=provider,
            zip=zip,
        )

        if not rename_requested and not version_updates_requested:
            raise _exceptions.PromptLayerValidationError(
                "Please provide at least one skill collection update.", response=None, body=None
            )

        skill_collection = None
        effective_identifier = identifier

        if rename_requested:
            rename_response = await aupdate_skill_collection(
                self.api_key,
                self.base_url,
                self.throw_on_error,
                identifier,
                {"name": name},
            )
            if not rename_response:
                return None
            skill_collection = rename_response["skill_collection"]
            effective_identifier = skill_collection["id"]

        version_response = None
        if version_updates_requested:
            save_payload = _build_version_payload(
                file_updates=file_updates,
                moves=moves,
                deletes=deletes,
                commit_message=commit_message,
                release_label=release_label,
                provider=provider,
            )
            if zip is None:
                version_response = await asave_skill_collection_version(
                    self.api_key,
                    self.base_url,
                    self.throw_on_error,
                    effective_identifier,
                    save_payload,
                )
            else:
                version_response = await asave_skill_collection_version(
                    self.api_key,
                    self.base_url,
                    self.throw_on_error,
                    effective_identifier,
                    save_payload,
                    zip=zip,
                )
            if not version_response:
                return None

        if skill_collection is None:
            pull_response = await self.pull(effective_identifier)
            if not pull_response or isinstance(pull_response, bytes):
                return None
            skill_collection = pull_response["skill_collection"]

        response: UpdateSkillCollectionResponse = {
            "success": True,
            "skill_collection": skill_collection,
        }
        if version_response:
            response["version"] = version_response["version"]
        return response
