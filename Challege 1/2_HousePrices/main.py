"""Chạy toàn bộ pipeline.

    python main.py                      # chạy mặc định
    python main.py --member minh        # gắn tên thành viên vào log
    python main.py --models lasso       # chỉ chạy 1 model
"""
import argparse
from src.pipeline.Pipeline import HousePricePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--member", default="team")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Mặc định lấy từ mục run.models trong config")
    args = ap.parse_args()
    out = HousePricePipeline(args.config, args.member).run(args.models)
    print(f"Submission saved to: {out}")


if __name__ == "__main__":
    main()
