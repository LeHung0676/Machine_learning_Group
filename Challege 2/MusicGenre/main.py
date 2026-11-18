"""Chạy toàn bộ pipeline Music Genre Classification.

    python main.py --member <ten>
    python main.py --config config/config_<ten>.yaml --member <ten>
    python main.py --models histgb logreg
"""
import argparse

from src.pipeline.Pipeline import MusicGenrePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--member", default="team")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Mặc định lấy từ mục run.models trong config")
    args = ap.parse_args()
    out = MusicGenrePipeline(args.config, args.member).run(args.models)
    print(f"\nSubmission saved to: {out}")


if __name__ == "__main__":
    main()
