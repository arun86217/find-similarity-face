from pathlib import Path
import cv2
import numpy as np
import insightface

MODEL_NAME = "buffalo_l"


def resolve_model_root() -> str:
    """
    InsightFace internally resolves models as:
        <root>/models/<name>

    So root must be the directory that CONTAINS "models".
    """

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
        "  ~/.insightface/models/buffalo_l\n\n"
        "Your system shows them at:\n"
        "  C:\\Users\\DELL\\.insightface\\models\\buffalo_l"
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

        # pre_start: loads models, will NOT download
        self.app.prepare(ctx_id=0, det_size=det_size)

    def get_face(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        faces = self.app.get(img)
        if not faces:
            raise RuntimeError(f"No face detected: {image_path}")

        # ✅ Always pick the largest detected face (best practice)
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        return faces[0]

    @staticmethod
    def similarity_percent(face1, face2) -> float:
        a = face1.normed_embedding
        b = face2.normed_embedding

        # cosine similarity already normalized in InsightFace
        sim = float(np.dot(a, b))

        # map [-1,1] → [0,100]
        return max(0.0, min(100.0, (sim + 1) * 50))
