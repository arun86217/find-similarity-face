import gradio as gr
import cv2
import numpy as np
from pathlib import Path

from face_engine import FaceEngine, crop_face, merge_faces

engine = FaceEngine()

# -----------------------------
# Utilities
# -----------------------------

def load_image_from_path(path):
    if not path:
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def files_to_gallery(files):
    images = []
    if not files:
        return images
    for f in files:
        img = cv2.imread(f.name)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


# -----------------------------
# Identify Mode
# -----------------------------

def identify_face(image_path, show_mark):
    if not image_path:
        return None, "[ERROR] No image uploaded\n"

    img = cv2.imread(image_path)
    faces = engine.app.get(img)

    if not faces:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "[ERROR] No face detected\n"

    logs = f"[OK] {len(faces)} face(s) detected\n"

    if show_mark:
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "FACE", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), logs


# -----------------------------
# Compare Mode
# -----------------------------

def compare_faces(source_path, target_files):
    if not source_path or not target_files:
        return "", "[ERROR] Source and targets required\n"

    logs = ""
    results = []

    src_name = Path(source_path).stem

    try:
        src = cv2.imread(source_path)
        src_face = engine.get_largest_face(src)
        logs += f"[OK] Source face detected ({src_name})\n"
    except Exception as e:
        return "", f"[ERROR] Source: {e}\n"

    for file_obj in target_files:
        name = Path(file_obj.name).name
        try:
            trg = cv2.imread(file_obj.name)
            if trg is None:
                raise RuntimeError("Invalid image")

            trg_face = engine.get_largest_face(trg)
            sim = engine.similarity_percent(src_face, trg_face)
            score = int(round(sim))

            results.append(f"{name} → {score}%")
            logs += f"[OK] {name} compared\n"

        except Exception as e:
            results.append(f"{name} → ERROR")
            logs += f"[ERROR] {name}: {e}\n"

    return "\n".join(results), logs


def save_face_pairs(source_path, target_files):
    if not source_path or not target_files:
        return "[ERROR] Source and targets required\n"

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    logs = ""
    src_name = Path(source_path).stem

    src = cv2.imread(source_path)
    src_face = engine.get_largest_face(src)
    src_crop = crop_face(src, src_face)

    for file_obj in target_files:
        trg_name = Path(file_obj.name).stem

        try:
            trg = cv2.imread(file_obj.name)
            if trg is None:
                raise RuntimeError("Invalid image")

            trg_face = engine.get_largest_face(trg)
            sim = engine.similarity_percent(src_face, trg_face)
            score = int(round(sim))

            trg_crop = crop_face(trg, trg_face)
            merged = merge_faces(src_crop, trg_crop)

            # ---- header zone (no face covered) ----
            h, w = merged.shape[:2]
            header_h = int(h * 0.22)
            canvas = np.ones((h + header_h, w, 3), dtype=np.uint8) * 255
            canvas[header_h:, :] = merged

            text = f"{score}%"
            scale = 2
            thick = 6
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            x = (w - tw) // 2
            y = int(header_h *1)

            cv2.putText(canvas, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)

            out_name = f"{src_name}_{trg_name}_{score}.jpg"
            cv2.imwrite(str(out_dir / out_name), canvas)

            logs += f"[SAVED] {out_name}\n"

        except Exception as e:
            logs += f"[ERROR] {trg_name}: {e}\n"

    return logs


# -----------------------------
# UI
# -----------------------------

with gr.Blocks(title="Face Similarity Tool") as demo:

    gr.Markdown("# 🧠 Face Similarity Tool")

    mode = gr.Radio(["Identify Face", "Compare Faces"],
                    value="Identify Face",
                    label="Mode")

    show_logs = gr.Checkbox(value=True, label="Show terminal log")
    logs = gr.Textbox(label="Terminal Log", lines=10, interactive=False, visible=True)

    # -------- Identify --------
    with gr.Column(visible=True) as identify_block:
        show_mark = gr.Checkbox(label="Show marked faces", value=True)
        with gr.Row():
            img_in = gr.Image(type="filepath", label="Upload image", height=420)
            out_img = gr.Image(label="Result image", height=420)
        id_btn = gr.Button("Submit")
        id_btn.click(identify_face, [img_in, show_mark], [out_img, logs])

    # -------- Compare --------
    with gr.Column(visible=False) as compare_block:

        with gr.Row():
            src_path = gr.Image(type="filepath", label="Source image", height=320)
            cmp_results = gr.Textbox(label="Similarity Results", lines=12)

            with gr.Column():
                trg_gallery = gr.Gallery(label="Target images", height=260, columns=3)
                trg_files = gr.File(file_types=["image"], file_count="multiple", label="Upload target images")

        trg_files.change(files_to_gallery, trg_files, trg_gallery)

        with gr.Row():
            cmp_btn = gr.Button("Compare")
            save_btn = gr.Button("Save cropped face pairs")

        cmp_btn.click(compare_faces, [src_path, trg_files], [cmp_results, logs])
        save_btn.click(save_face_pairs, [src_path, trg_files], logs)

    # -------- Mode switch --------
    def switch_mode(m):
        return (
            gr.update(visible=(m == "Identify Face")),
            gr.update(visible=(m == "Compare Faces"))
        )

    mode.change(switch_mode, mode, [identify_block, compare_block])
    show_logs.change(lambda x: gr.update(visible=x), show_logs, logs)

demo.launch(debug=True)
