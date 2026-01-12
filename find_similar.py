#!/usr/bin/env python3

import argparse
import os
from time import time
from face_engine import FaceEngine


def main():
    parser = argparse.ArgumentParser(description="Simple face similarity tool")
    parser.add_argument("--source", required=True, help="Source face image")
    parser.add_argument("--target", required=True, help="Comma separated target images")

    args = parser.parse_args()

    engine = FaceEngine()

    source_face = engine.get_face(args.source)
    targets = [x.strip() for x in args.target.split(",") if x.strip()]

    for t in targets:
        try:
            target_face = engine.get_face(t)
            percent = engine.similarity_percent(source_face, target_face)

            print(
                f"{os.path.basename(args.source)} and "
                f"{os.path.basename(t)} has a similarity of {percent:.0f}%"
            )

        except Exception as e:
            print(f"[ERROR] {t}: {e}")


if __name__ == "__main__":
    start = time()
    main()
    print(f'Total Time took for comparison : {round(time() - start)} seconds')
