import re
from datetime import datetime
from uuid import UUID


class Any:
    def __init__(self, *, type_=None, regex=None):
        self.type = type_
        self.regex = regex if isinstance(regex, (re.Pattern, type(None))) else re.compile(regex)

    def __eq__(self, other):
        result = True
        if (type_ := self.type) is not None:
            result &= isinstance(other, type_)
        if regex := self.regex:
            result &= bool(regex.match(str(other)))

        return result

    def __ne__(self, other):
        return self != other

    def __repr__(self):
        return (
            f"Any(type_={'None' if self.type is None else self.type.__name__}, "
            f"regex={'None' if self.regex is None else self.regex.pattern})"
        )

    def __str__(self):
        return repr(self)


ANY_INT = Any(type_=int)
ANY_BOOL = Any(type_=bool)
ANY_STR = Any(type_=str)
ANY_DATETIME = Any(type_=datetime)
ANY_UUID = Any(type_=UUID)
