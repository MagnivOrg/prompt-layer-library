from .base import Base


class Prompt(Base):
    BASE_URL = Base.BASE_URL + "/prompts"

    @classmethod
    def list(cls, params={}):
        """
        List all prompts
        """
        return super().list(params)
