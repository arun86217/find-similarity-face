# Face Similarity

A minimal CLI tool to compare two or more faces using InsightFace.

## Install

python -m venv venv  
source venv/bin/activate  (or venv\Scripts\activate)  

pip install -r requirements.txt

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

├── .gitignore
├── README.md
├── face_engine.py
├── find_similar.py
├── models
│   ├── 1k3d68.onnx
│   ├── 2d106det.onnx
│   ├── buffalo_l.zip
│   ├── det_10g.onnx
│   ├── genderage.onnx
│   └── w600k_r50.onnx
├── requirements.txt
└── storage
    ├── a.jpg
    ├── b.jpg
    ├── c.jpg
    ├── d.jpg
    └── f.jpg