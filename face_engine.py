#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import insightface

MODEL_NAME = "buffalo_l"


def resolve_model_root() -> str:
    # 1) Project-local models (./models/buffalo_l)
    local_root = Path.cwd()
    if (local_root / "models" / MODEL_NAME).exists():
        return str(local_root)

    # 2) Global InsightFace cache (~/.insightface/models/buffalo_l)
    home_root = Path.home() / ".insightface"
    if (home_root / "models" / MODEL_NAME).exists():
        return str(home_root)

    raise RuntimeError(
        "InsightFace models not found.\n"
        "Expected one of:\n"
        "  ./models/buffalo_l\n"
        "  ~/.insightface/models/buffalo_l"
    )
class FaceEngine:
    def __init__(self, providers=None, det_size=(640, 640)):
        model_root = resolve_model_root()
        print(f"[INFO] Using InsightFace root: {model_root}")

        self.app = insightface.app.FaceAnalysis(
            name=MODEL_NAME,
            root=model_root,
            providers=providers or ["CPUExecutionProvider"],
        )

        self.app.prepare(ctx_id=0, det_size=det_size)

    # ---------- core detection ----------
    def _get_largest_face_from_image(self, image):
        faces = self.app.get(image)
        if not faces:
            return None

        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces[0]

    # ---------- public APIs ----------
    def get_face(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        face = self._get_largest_face_from_image(img)
        if face is None:
            raise RuntimeError(f"No face detected: {image_path}")

        return face

    def get_largest_face(self, image):
        face = self._get_largest_face_from_image(image)
        if face is None:
            raise RuntimeError("No face detected")
        return face

    @staticmethod
    def similarity_percent(face1, face2) -> float:
        a = face1.normed_embedding
        b = face2.normed_embedding
        sim = float(np.dot(a, b))
        return max(0.0, min(100.0, (sim + 1) * 50))

def draw_face_box(image, face, label="FACE"):
    x1, y1, x2, y2 = map(int, face.bbox)

    # draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)

    # label text
    cv2.putText(
        image,
        label,
        (x1 + 3, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return image

def crop_face(image, face, margin=0.35):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, face.bbox)

    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * margin)
    my = int(bh * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    return image[y1:y2, x1:x2]

def merge_faces(face1_img, face2_img, size=320):
    face1_img = cv2.resize(face1_img, (size, size))
    face2_img = cv2.resize(face2_img, (size, size))
    return np.hstack([face1_img, face2_img])

def main():
    parser = argparse.ArgumentParser(description="Face detector with bounding box output")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--show", action="store_true", help="Save image with detected face box")

    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    engine = FaceEngine()

    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError("Failed to load image")

    face = engine.get_largest_face(image)
    if face is None:
        raise RuntimeError("No face detected")

    # InsightFace detection confidence
    score = getattr(face, "det_score", 1.0) * 100
    label = f"FACE {score:.0f}%"

    result = draw_face_box(image, face, label)

    out_path = img_path.with_name(img_path.stem + "_show" + img_path.suffix)
    cv2.imwrite(str(out_path), result)

    print(f"[OK] Face image saved to: {out_path}")


if __name__ == "__main__":
    main()
