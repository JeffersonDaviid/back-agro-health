from typing import List

from pydantic import BaseModel


class DataPoint(BaseModel):
    year: int
    month: int
    day: int
    Latitude: float
    Longitude: float


class InputData(BaseModel):
    data: List[DataPoint]
