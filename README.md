# AI Internship Tasks

Portfolio of the tasks completed during my Artificial Intelligence internship, one task per week.

**Author:** Fiza Asghar

## Tasks

| # | Task | Topic | Folder |
|---|------|-------|--------|
| 1 | LinkedIn announcement | Professional branding: post about the internship offer | [`task1-linkedin-post`](task1-linkedin-post/) |
| 2 | Intelligent Multi-Class Text Sentiment Classifier | NLP, TF-IDF, Logistic Regression / SVM, RoBERTa fine-tuning | [`task2-sentiment-classifier`](task2-sentiment-classifier/) |
| 3 | Heuristic Graph Pathfinding Agent Search Engine | A* and Dijkstra from scratch, grid mazes, search metrics | [`task3-pathfinding-agent`](task3-pathfinding-agent/) |
| 4 | Computer Vision Object Detector & Frame-by-Frame Segmenter Pipeline | OpenCV contours and tracking, YOLOv8, performance whitepaper | [`task4-cv-pipeline`](task4-cv-pipeline/) |

## Highlights

**Task 2: Sentiment classifier** (Twitter US Airline Sentiment, 3 classes)

| Version | Approach | Accuracy | Macro F1 |
|---|---|---|---|
| v1 | TF-IDF + Logistic Regression | 77.2% | 0.725 |
| v2 | TF-IDF (word + char) + tuned SVM, negation handling | 82.0% | 0.770 |
| v3 | Fine-tuned RoBERTa | **86.0%** | **0.822** |

**Task 3: Pathfinding agent** (300 runs on random grids and mazes)

| Metric (mean over solved grids) | Dijkstra | A* |
|---|---|---|
| Expanded nodes | 1259.7 | **334.4** (73.5% fewer) |
| Runtime | 3.71 ms | **1.11 ms** |
| Path cost | optimal | optimal (identical) |

**Task 4: Computer vision pipelines**

| Pipeline | Test | Result |
|---|---|---|
| A: OpenCV (Gaussian + adaptive threshold + contours + tracking) | Synthetic video with ground truth | Macro F1 **0.990**, about 248 FPS on a CPU |
| B: YOLOv8n (ByteTrack) | 650-frame room video, Tesla T4 | about 40 FPS, detections in 35% of frames, with confident mistakes analysed in the whitepaper |

## Tech stack

Python, NumPy, pandas, scikit-learn, NLTK, PyTorch, Hugging Face Transformers, OpenCV, Ultralytics YOLO, Matplotlib, Google Colab.

## Notes

- Each task folder has its own README with methodology, results and run instructions, and a PDF report.
- Datasets and personal videos are not included; the Task 2 README links to the Kaggle source.
