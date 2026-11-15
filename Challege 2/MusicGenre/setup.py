"""Cài dự án thành package:  pip install -e .
Sau đó `import src.*` chạy được từ bất cứ thư mục nào."""
from setuptools import find_packages, setup

setup(
    name="music-genre",
    version="0.1.0",
    description="Music Genre Classification - Nhóm 14",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
