import os
from typing import Type

import requests
from pydantic import BaseModel, ValidationError


class Base:
    API_KEY = os.environ.get("PROMPTLAYER_API_KEY", "")
    BASE_URL = os.environ.get("BASE_URL", "https://api.promptlayer.com/rest")

    @classmethod
    def list(cls, params_model: Type[BaseModel], params={}):
        """
        List all resources
        """
        try:
            params = params_model.parse_obj(params).dict()
        except ValidationError as e:
            raise Exception(f"Invalid pagination arguments: {e}")

        # TODO: parse the response into a list of pydantic objects
        return requests.get(
            cls.BASE_URL,
            headers={"X-API-KEY": cls.API_KEY},
            params=params,
        ).json()
