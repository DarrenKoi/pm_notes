from pydantic import BaseModel


class KnowhowItem(BaseModel):
    knowhow_no: int
    KNOWHOW_ID: str = ""
    knowhow: str
    user_id: str
    user_name: str
    user_department: str


class EnrichedKnowhow(KnowhowItem):
    summary: str = ""
    category: str = ""
    keywords: list[str] = []


class KnowhowFile(BaseModel):
    data: list[KnowhowItem]
