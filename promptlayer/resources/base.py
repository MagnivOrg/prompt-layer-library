import os

import requests

from promptlayer.utils import get_api_key


class Base:
    API_KEY = get_api_key()
    BASE_URL = os.environ.get("BASE_URL", "https://api.promptlayer.com/rest")

    @classmethod
    def list(cls, params={}):
        """
        List all resources
        """
        return requests.get(
            cls.BASE_URL,
            headers={"X-API-KEY": cls.API_KEY},
            params=params,
        ).json()
