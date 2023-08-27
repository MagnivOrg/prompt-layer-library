import os

import requests


class Base:
    API_KEY = os.environ.get("PROMPTLAYER_API_KEY", "")
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
