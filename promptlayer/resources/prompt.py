from promptlayer.schemas import pagination

from .base import Base


class Prompt(Base):
    BASE_URL = Base.BASE_URL + "/prompts"

    @classmethod
    def list(cls, params={}):
        """
        List all prompts
        """
        # TODO: parse the response into a list of prompts
        return super().list(pagination.Base, params)
