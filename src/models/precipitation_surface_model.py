from typing import List

from pydantic import BaseModel


class DataPointPrecipit(BaseModel):
    ConvectivePrecip: float
    ProbabilityofPrecip: int
    SunglintAngle: int
    Temp2Meter: float
    TotalColWaterVapor: int
    Latitude: float
    Longitude: float


class InputDataPrecipit(BaseModel):
    data: List[DataPointPrecipit]
