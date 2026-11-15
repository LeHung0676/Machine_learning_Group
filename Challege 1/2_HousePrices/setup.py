"""Cài dự án thành package để import được từ bất cứ đâu.

    pip install -e .

Sau bước này, `import src.model.Lasso` chạy được dù bạn đang ở notebooks/,
tests/ hay thư mục nào khác, nên không cần sys.path.append nữa.
"""
from setuptools import find_packages, setup

setup(
    name="house-prices",
    version="0.1.0",
    description="House Prices Prediction - ML Group (4 thành viên)",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
