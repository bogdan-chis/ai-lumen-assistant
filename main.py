from src.app import run
import argparse
import os
import sys

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Process a video file for OCR/Q&A.")
    p.add_argument("video_path", help="Path to an .mp4 or other readable video file")
    p.add_argument("--no-gui", action="store_true", help="Disable preview window")
    args = p.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    run(video_path=args.video_path, visualize=not args.no_gui)
