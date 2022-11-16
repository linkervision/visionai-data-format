from pydantic import BaseModel


class Image(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    coco_url: str


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    iscrowd: int


class Category(BaseModel):
    id: int
    name: str


class Info(BaseModel):
    year: str = ""
    version: str = ""
    description: str = ""
    contributor: str = ""
    url: str = ""
    date_created: str = ""


class Coco(BaseModel):
    info: Info = Info()
    licenses: list = []
    categories: list[Category]
    images: list[Image]
    annotations: list[Annotation]
