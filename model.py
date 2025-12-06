from pydantic import BaseModel, Field

class Response(BaseModel):
    headline: str = Field(..., description="The main headline of the response")
    SubHeadline: str = Field(..., description="The sub headline of the in the brief")
    point1 : str = Field(..., description="The first key point in the brief")
    point2 : str = Field(..., description="The second key point in the brief")
    point3 : str = Field(..., description="The third key point in the brief")
    summary: str = Field(..., description="A concise summary of the response")