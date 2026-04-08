import json
import os
from pathlib import Path
from typing import IO, Any, Dict, Union
from urllib.parse import quote

import httpx
import requests

from promptlayer import exceptions as _exceptions
from promptlayer.types.skill import (
    CreateSkillCollection,
    CreateSkillCollectionResponse,
    PullSkillCollectionResponse,
    SaveSkillCollectionVersion,
    UpdateSkillCollection,
)
from promptlayer.utils import (
    _get_requests_session,
    _make_httpx_client,
    logger,
    raise_on_bad_response,
    retry_on_api_error,
    warn_on_bad_response,
)

ZipUpload = Union[bytes, bytearray, str, os.PathLike[str], IO[bytes]]


def _skill_collection_endpoint(base_url: str, identifier: str) -> str:
    return f"{base_url}/api/public/v2/skill-collections/{quote(identifier, safe='')}"


def _build_pull_skill_collection_params(
    *,
    label: Union[str, None] = None,
    version: Union[int, None] = None,
    format: Union[str, None] = None,
) -> Dict[str, Any]:
    params = {}
    if label is not None:
        params["label"] = label
    if version is not None:
        params["version"] = version
    if format is not None:
        params["format"] = format
    return params


def _build_zip_upload_file(zip_file: ZipUpload):
    if isinstance(zip_file, (bytes, bytearray)):
        return ("skills.zip", bytes(zip_file), "application/zip")

    if isinstance(zip_file, (str, os.PathLike)):
        path = Path(zip_file)
        return (path.name or "skills.zip", path.read_bytes(), "application/zip")

    if hasattr(zip_file, "read"):
        content = zip_file.read()
        if isinstance(content, str):
            raise TypeError("Zip file-like objects must return bytes, not text.")
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("Zip file-like objects must return bytes.")
        filename = Path(getattr(zip_file, "name", "skills.zip")).name or "skills.zip"
        return (filename, bytes(content), "application/zip")

    raise TypeError("zip must be bytes, a filesystem path, or a binary file-like object.")


@retry_on_api_error
def pull_skill_collection(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    identifier: str,
    *,
    label: Union[str, None] = None,
    version: Union[int, None] = None,
    format: Union[str, None] = None,
) -> Union[PullSkillCollectionResponse, bytes, None]:
    try:
        response = _get_requests_session().get(
            _skill_collection_endpoint(base_url, identifier),
            headers={"X-API-KEY": api_key},
            params=_build_pull_skill_collection_params(label=label, version=version, format=format),
        )
        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while pulling your skill collection",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while pulling your skill collection",
                )
                return None
        if format == "zip":
            return response.content
        return response.json()
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while pulling your skill collection: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while pulling your skill collection: {e}")
        return None


@retry_on_api_error
async def apull_skill_collection(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    identifier: str,
    *,
    label: Union[str, None] = None,
    version: Union[int, None] = None,
    format: Union[str, None] = None,
) -> Union[PullSkillCollectionResponse, bytes, None]:
    try:
        async with _make_httpx_client() as client:
            response = await client.get(
                _skill_collection_endpoint(base_url, identifier),
                headers={"X-API-KEY": api_key},
                params=_build_pull_skill_collection_params(label=label, version=version, format=format),
            )
        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while pulling your skill collection",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while pulling your skill collection",
                )
                return None
        if format == "zip":
            return response.content
        return response.json()
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while pulling your skill collection: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while pulling your skill collection: {e}")
        return None


@retry_on_api_error
def create_skill_collection(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    body: CreateSkillCollection,
    zip: Union[ZipUpload, None] = None,
) -> Union[CreateSkillCollectionResponse, None]:
    try:
        request_kwargs = {
            "headers": {"X-API-KEY": api_key},
        }
        if zip is None:
            request_kwargs["json"] = body
        else:
            request_kwargs["data"] = {"metadata": json.dumps(body)}
            request_kwargs["files"] = {"zip": _build_zip_upload_file(zip)}
        response = _get_requests_session().post(
            f"{base_url}/api/public/v2/skill-collections",
            **request_kwargs,
        )
        if response.status_code != 201:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while creating your skill collection",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while creating your skill collection",
                )
                return None
        return response.json()
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while creating your skill collection: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while creating your skill collection: {e}")
        return None


@retry_on_api_error
async def acreate_skill_collection(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    body: CreateSkillCollection,
    zip: Union[ZipUpload, None] = None,
) -> Union[CreateSkillCollectionResponse, None]:
    try:
        request_kwargs = {
            "headers": {"X-API-KEY": api_key},
        }
        if zip is None:
            request_kwargs["json"] = body
        else:
            request_kwargs["data"] = {"metadata": json.dumps(body)}
            request_kwargs["files"] = {"zip": _build_zip_upload_file(zip)}
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{base_url}/api/public/v2/skill-collections",
                **request_kwargs,
            )
        if response.status_code != 201:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while creating your skill collection",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while creating your skill collection",
                )
                return None
        return response.json()
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while creating your skill collection: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while creating your skill collection: {e}")
        return None


@retry_on_api_error
def update_skill_collection(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    identifier: str,
    body: UpdateSkillCollection,
) -> Union[Dict[str, Any], None]:
    try:
        response = _get_requests_session().patch(
            _skill_collection_endpoint(base_url, identifier),
            headers={"X-API-KEY": api_key},
            json=body,
        )
        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while updating your skill collection",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while updating your skill collection",
                )
                return None
        return response.json()
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while updating your skill collection: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while updating your skill collection: {e}")
        return None


@retry_on_api_error
async def aupdate_skill_collection(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    identifier: str,
    body: UpdateSkillCollection,
) -> Union[Dict[str, Any], None]:
    try:
        async with _make_httpx_client() as client:
            response = await client.patch(
                _skill_collection_endpoint(base_url, identifier),
                headers={"X-API-KEY": api_key},
                json=body,
            )
        if response.status_code != 200:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while updating your skill collection",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while updating your skill collection",
                )
                return None
        return response.json()
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while updating your skill collection: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while updating your skill collection: {e}")
        return None


@retry_on_api_error
def save_skill_collection_version(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    identifier: str,
    body: SaveSkillCollectionVersion,
    zip: Union[ZipUpload, None] = None,
) -> Union[Dict[str, Any], None]:
    try:
        request_kwargs = {
            "headers": {"X-API-KEY": api_key},
        }
        if zip is None:
            request_kwargs["json"] = body
        else:
            request_kwargs["data"] = {"metadata": json.dumps(body)}
            request_kwargs["files"] = {"zip": _build_zip_upload_file(zip)}
        response = _get_requests_session().post(
            f"{_skill_collection_endpoint(base_url, identifier)}/versions",
            **request_kwargs,
        )
        if response.status_code != 201:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while saving your skill collection version",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while saving your skill collection version",
                )
                return None
        return response.json()
    except requests.exceptions.RequestException as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while saving your skill collection version: {e}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while saving your skill collection version: {e}")
        return None


async def asave_skill_collection_version(
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    identifier: str,
    body: SaveSkillCollectionVersion,
    zip: Union[ZipUpload, None] = None,
) -> Union[Dict[str, Any], None]:
    try:
        request_kwargs = {
            "headers": {"X-API-KEY": api_key},
        }
        if zip is None:
            request_kwargs["json"] = body
        else:
            request_kwargs["data"] = {"metadata": json.dumps(body)}
            request_kwargs["files"] = {"zip": _build_zip_upload_file(zip)}
        async with _make_httpx_client() as client:
            response = await client.post(
                f"{_skill_collection_endpoint(base_url, identifier)}/versions",
                **request_kwargs,
            )
        if response.status_code != 201:
            if throw_on_error:
                raise_on_bad_response(
                    response,
                    "PromptLayer had the following error while saving your skill collection version",
                )
            else:
                warn_on_bad_response(
                    response,
                    "WARNING: PromptLayer had the following error while saving your skill collection version",
                )
                return None
        return response.json()
    except httpx.RequestError as e:
        if throw_on_error:
            raise _exceptions.PromptLayerAPIConnectionError(
                f"PromptLayer had the following error while saving your skill collection version: {str(e)}",
                response=None,
                body=None,
            ) from e
        logger.warning(f"PromptLayer had the following error while saving your skill collection version: {e}")
        return None
