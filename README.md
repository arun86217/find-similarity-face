# Face Similarity

A lightweight CLI tool to compare one face against one or more target faces using InsightFace.

This project is focused only on **face detection + face similarity** .

---

## Features

- Face detection and recognition (InsightFace)
- Multiple target images support
- Automatically selects the largest face in each image
- Works with full-body images (face is detected internally)
- Offline support when models are available locally
- Simple, scriptable CLI

---

## Install

### 1. Create environment

```bash
python -m venv venv  
source venv/bin/activate  (or venv\Scripts\activate)  

pip install -r requirements.txt
```

Models are stored locally in ./models/

On first run, required InsightFace models are downloaded automatically.

Make sure you have internet access the first time you run the tool.

## Usage

python find_similar.py --source s.jpg --target j.jpg

python find_similar.py --source s.jpg --target j.jpg,g.jpg

## Output

s.jpg and j.jpg has a similarity of 66%
s.jpg and g.jpg has a similarity of 16%

## Project Structure

find_similarity_face/
├── .gitignore
├── README.md
├── face_engine.py
├── find_similar.py
├── requirements.txt
├── models/            # optional, not committed
│   └── buffalo_l/
│       ├── 1k3d68.onnx
│       ├── 2d106det.onnx
│       ├── buffalo_l.zip
│       ├── det_10g.onnx
│       ├── genderage.onnx
│       └── w600k_r50.onnx
└── storage/           # sample images (optional, not committed)
    ├── a.jpg
    ├── b.jpg
    ├── c.jpg
    ├── d.jpg
    └── f.jpg