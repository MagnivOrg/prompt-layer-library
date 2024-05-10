from promptlayer.groups.groups import create


class GroupManager:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create(self):
        return create(self.api_key)


__all__ = ["GroupManager"]
