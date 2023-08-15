from typing import Generic, TypeVar, Union, List

from pydantic import BaseModel, PositiveInt


class Base(BaseModel):
    page: PositiveInt
    per_page: PositiveInt


Item = TypeVar("Item", bound=BaseModel)


class Response(Base, Generic[Item]):
    has_next: bool
    has_prev: bool
    next_num: Union[PositiveInt, None]
    prev_num: Union[PositiveInt, None]
    page: PositiveInt
    total: PositiveInt
    items: List[Item]
