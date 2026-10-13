"""
Task 4: Computer Vision Object Detector & Frame-by-Frame Segmenter Pipeline  (Part A: classical OpenCV)
---------------------------------------------------------------------------------------------------
Real-time pipeline that extracts and tracks target contours in a video stream:

  frame -> grayscale -> Gaussian blur -> adaptive threshold -> morphology
        -> contours -> pixel-bound (area) filter -> shape classification
        -> centroid tracking -> overlay of per-class count vector + FPS

Every frame is timed stage by stage. If the input is the built-in synthetic video, the exact ground
truth is known, so detection quality (precision / recall / F1, count error, ID switches) and a small
ablation study (no blur, global Otsu threshold, no area bounds) are computed as well.

Run:
    python cv_pipeline.py                         # synthetic video + full evaluation (no camera needed)
    python cv_pipeline.py --source my_video.mp4   # your own video (metrics without ground truth)
    python cv_pipeline.py --source 0 --show       # live webcam, press q to quit
Needs:  pip install opencv-python numpy pandas matplotlib
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHAPES = ["circle", "triangle", "rectangle"]          # target classes
PRED_CLASSES = SHAPES + ["other"]
BGR = {"circle": (60, 160, 255), "triangle": (80, 220, 80), "rectangle": (255, 150, 60), "other": (180, 180, 180)}


# ================================================================== parameters
@dataclass
class Params:
    blur_ksize: int = 5          # Gaussian kernel size (odd)
    block_size: int = 41         # adaptive threshold neighbourhood (odd)
    c_const: int = 10            # constant subtracted from the local Gaussian-weighted mean
    morph_ksize: int = 3         # kernel for morphological close
    min_area: int = 250          # pixel bounds: contours with fewer pixels are noise
    max_area: int = 20000        #               contours with more pixels are background / merged blobs
    approx_eps: float = 0.045    # polygon approximation tolerance (fraction of perimeter)
    circ_thr: float = 0.84       # circularity 4*pi*A/P^2 above which a contour is called a circle
    track_dist: float = 50.0     # max centroid jump (px) between frames for the same object
    track_missing: int = 5       # frames a track may be lost before it is deleted


# ================================================================ synthetic video
def make_synthetic_video(path, n_frames=300, w=640, h=480, n_shapes=7, seed=0, fps=30):
    """Moving circles / triangles / rectangles on an unevenly lit, noisy background, plus speckle noise.
    Returns the ground truth: for every frame a list of {id, cls, cx, cy}."""
    rng = np.random.default_rng(seed)
    palette = [(160, 50, 30), (40, 110, 40), (40, 40, 170), (130, 40, 120), (30, 90, 150)]
    objs = []
    for i in range(n_shapes):
        objs.append(dict(id=i, cls=SHAPES[i % 3], r=int(rng.integers(20, 34)),
                         x=float(rng.uniform(60, w - 60)), y=float(rng.uniform(60, h - 60)),
                         vx=float(rng.uniform(-3, 3)), vy=float(rng.uniform(-3, 3)),
                         ang=float(rng.uniform(0, 6.28)), spin=float(rng.uniform(-0.05, 0.05)),
                         color=palette[i % len(palette)], aspect=float(rng.uniform(1.0, 1.6))))
    # uneven illumination: left side darker, right side brighter
    xx = np.linspace(150, 235, w, dtype=np.float32)[None, :].repeat(h, 0)
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    gt = []
    for _ in range(n_frames):
        frame = np.repeat(xx[:, :, None], 3, axis=2) + rng.normal(0, 7, (h, w, 3)).astype(np.float32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame_gt = []
        for o in objs:
            o["x"] += o["vx"]; o["y"] += o["vy"]; o["ang"] += o["spin"]
        for i, a in enumerate(objs):                          # shapes bounce off each other (no overlap)
            for b in objs[i + 1:]:
                dx, dy = b["x"] - a["x"], b["y"] - a["y"]
                dist = math.hypot(dx, dy) or 1e-6
                gap = 1.3 * (a["r"] + b["r"]) + 14
                if dist < gap:
                    nx, ny, push = dx / dist, dy / dist, (gap - dist) / 2
                    a["x"] -= nx * push; a["y"] -= ny * push; b["x"] += nx * push; b["y"] += ny * push
                    a["vx"], b["vx"], a["vy"], b["vy"] = b["vx"], a["vx"], b["vy"], a["vy"]
        for o in objs:
            m = o["r"] + 5
            if not m < o["x"] < w - m: o["vx"] *= -1; o["x"] = float(np.clip(o["x"], m, w - m))
            if not m < o["y"] < h - m: o["vy"] *= -1; o["y"] = float(np.clip(o["y"], m, h - m))
            c, r = (o["x"], o["y"]), o["r"]
            if o["cls"] == "circle":
                cv2.circle(frame, (int(c[0]), int(c[1])), r, o["color"], -1, cv2.LINE_AA)
            elif o["cls"] == "rectangle":
                box = cv2.boxPoints(((c[0], c[1]), (2 * r * 0.9, 2 * r * 0.9 / o["aspect"]), math.degrees(o["ang"])))
                cv2.fillPoly(frame, [box.astype(np.int32)], o["color"], cv2.LINE_AA)
            else:
                pts = np.array([[c[0] + 1.25 * r * math.cos(o["ang"] + k * 2.094),
                                 c[1] + 1.25 * r * math.sin(o["ang"] + k * 2.094)] for k in range(3)])
                cv2.fillPoly(frame, [pts.astype(np.int32)], o["color"], cv2.LINE_AA)
            frame_gt.append(dict(id=o["id"], cls=o["cls"], cx=c[0], cy=c[1]))
        for _ in range(20):                                   # speckle noise (should be removed by the area bound)
            cv2.circle(frame, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 4)), (50, 50, 50), -1)
        wr.write(frame)
        gt.append(frame_gt)
    wr.release()
    return gt


# ======================================================================= tracker
class CentroidTracker:
    """Greedy nearest-centroid tracker that gives every contour a persistent ID and keeps a trail."""

    def __init__(self, max_dist, max_missing):
        self.max_dist, self.max_missing = max_dist, max_missing
        self.tracks, self.next_id, self.created = {}, 0, 0

    def update(self, dets):
        pairs = sorted((math.dist(t["c"], (d["cx"], d["cy"])), tid, di)
                       for tid, t in self.tracks.items() for di, d in enumerate(dets))
        used_t, used_d = set(), set()
        for dist, tid, di in pairs:
            if dist > self.max_dist or tid in used_t or di in used_d:
                continue
            t = self.tracks[tid]
            t["c"], t["missing"] = (dets[di]["cx"], dets[di]["cy"]), 0
            t["trail"].append(t["c"]); t["trail"] = t["trail"][-25:]
            dets[di]["id"] = tid
            used_t.add(tid); used_d.add(di)
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]
        for di, d in enumerate(dets):
            if di not in used_d:
                c = (d["cx"], d["cy"])
                self.tracks[self.next_id] = dict(c=c, missing=0, trail=[c])
                d["id"] = self.next_id
                self.next_id += 1; self.created += 1
        return dets


# ================================================================== per-frame pipeline
def classify(cnt, P):
    area, peri = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
    if peri == 0:
        return None
    circ = 4 * math.pi * area / (peri * peri)
    v = len(cv2.approxPolyDP(cnt, P.approx_eps * peri, True))
    if circ >= P.circ_thr: cls = "circle"
    elif v == 3: cls = "triangle"
    elif v == 4: cls = "rectangle"
    else: cls = "other"
    return cls, area, circ, v


def process_frame(frame, P, thresh="adaptive", blur=True, keep_debug=False):
    """Returns (detections, stage timings in ms, debug images)."""
    T, dbg = {}, {}
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t1 = time.perf_counter(); T["gray"] = (t1 - t0) * 1e3
    smooth = cv2.GaussianBlur(gray, (P.blur_ksize, P.blur_ksize), 0) if blur else gray
    t2 = time.perf_counter(); T["blur"] = (t2 - t1) * 1e3
    if thresh == "adaptive":
        binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                       P.block_size, P.c_const)
    else:                                                      # global Otsu threshold (ablation)
        _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P.morph_ksize, P.morph_ksize))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    t3 = time.perf_counter(); T["threshold"] = (t3 - t2) * 1e3
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if not (P.min_area <= a <= P.max_area):                # pixel bounds
            continue
        res = classify(cnt, P)
        if res is None:
            continue
        cls, area, circ, v = res
        m = cv2.moments(cnt)
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (0, 0)
        dets.append(dict(cls=cls, area=area, circ=circ, vertices=v, cx=cx, cy=cy,
                         bbox=cv2.boundingRect(cnt), contour=cnt))
    t4 = time.perf_counter(); T["contours"] = (t4 - t3) * 1e3
    if keep_debug:
        dbg = dict(gray=gray, smooth=smooth, binary=binary)
    return dets, T, dbg


# ==================================================================== overlay
def draw_overlay(frame, dets, tracker, counts, fps, latency_ms):
    out = frame.copy()
    for d in dets:
        col = BGR[d["cls"]]
        for tid_trail in [tracker.tracks.get(d.get("id"), {}).get("trail", [])]:
            if len(tid_trail) > 1:
                cv2.polylines(out, [np.array(tid_trail, np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.drawContours(out, [d["contour"]], -1, col, 2, cv2.LINE_AA)
        x, y, w, h = d["bbox"]
        cv2.putText(out, f'#{d.get("id", "?")} {d["cls"]}', (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    # HUD with the dynamic count vector
    panel = out.copy()
    cv2.rectangle(panel, (0, 0), (360, 78), (0, 0, 0), -1)
    out = cv2.addWeighted(panel, 0.55, out, 0.45, 0)
    vec = ", ".join(str(counts[c]) for c in SHAPES)
    cv2.putText(out, f"count [circle, triangle, rectangle] = [{vec}]", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"total objects: {sum(counts.values())}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS {fps:6.1f}   latency {latency_ms:5.2f} ms", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ================================================================== evaluation
def new_confusion():
    # rows: GT circle, triangle, rectangle, background(false alarm); cols: pred circle, triangle, rectangle, other, missed
    return np.zeros((len(SHAPES) + 1, len(PRED_CLASSES) + 1), dtype=int)


def evaluate_frame(gt, dets, conf, id_state, max_d=22):
    pairs = sorted((math.dist((g["cx"], g["cy"]), (d["cx"], d["cy"])), gi, di)
                   for gi, g in enumerate(gt) for di, d in enumerate(dets))
    ug, ud = set(), set()
    for dist, gi, di in pairs:
        if dist > max_d or gi in ug or di in ud:
            continue
        ug.add(gi); ud.add(di)
        conf[SHAPES.index(gt[gi]["cls"]), PRED_CLASSES.index(dets[di]["cls"])] += 1
        gid, tid = gt[gi]["id"], dets[di].get("id")
        if gid in id_state["map"] and id_state["map"][gid] != tid:
            id_state["switches"] += 1
        id_state["map"][gid] = tid
    for gi, g in enumerate(gt):
        if gi not in ug:
            conf[SHAPES.index(g["cls"]), -1] += 1
    for di, d in enumerate(dets):
        if di not in ud:
            conf[-1, PRED_CLASSES.index(d["cls"])] += 1


def scores_from_confusion(conf):
    out = {}
    for i, c in enumerate(SHAPES):
        tp = conf[i, i]
        fp, fn = conf[:, i].sum() - tp, conf[i, :].sum() - tp
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[c] = dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), tp=int(tp), fp=int(fp), fn=int(fn))
    out["macro_f1"] = float(np.mean([out[c]["f1"] for c in SHAPES]))
    matched = conf[:3, :4].sum()
    out["localisation_precision"] = float(matched / max(matched + conf[3, :4].sum(), 1))
    out["localisation_recall"] = float(matched / max(matched + conf[:3, 4].sum(), 1))
    return out


def run_variant(video_path, gt, P, thresh="adaptive", blur=True):
    """Runs the pipeline on the whole video without drawing, and returns quality + speed metrics."""
    cap = cv2.VideoCapture(video_path)
    conf, lat, tr = new_confusion(), [], CentroidTracker(P.track_dist, P.track_missing)
    id_state = dict(map={}, switches=0)
    for f in range(len(gt)):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dets, _, _ = process_frame(frame, P, thresh, blur)
        tr.update(dets)
        lat.append((time.perf_counter() - t0) * 1e3)
        evaluate_frame(gt[f], dets, conf, id_state)
    cap.release()
    s = scores_from_confusion(conf)
    return dict(macro_f1=s["macro_f1"], loc_precision=s["localisation_precision"], loc_recall=s["localisation_recall"],
                mean_latency_ms=float(np.mean(lat)), fps=float(1000 / np.mean(lat)), id_switches=id_state["switches"])


# ===================================================================== figures
def make_figures(figdir, rows, stage_img, conf, ablation, has_gt):
    os.makedirs(figdir, exist_ok=True)
    df = pd.DataFrame(rows)

    # 1. pipeline stages
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.6))
    titles = ["1. Original frame", "2. Grayscale", "3. Gaussian blur", "4. Adaptive threshold + morphology",
              "5. Contours after pixel-bound filter + classification", "6. Final overlay (IDs, trails, count vector)"]
    imgs = [cv2.cvtColor(stage_img["frame"], cv2.COLOR_BGR2RGB), stage_img["gray"], stage_img["smooth"], stage_img["binary"],
            cv2.cvtColor(stage_img["contours"], cv2.COLOR_BGR2RGB), cv2.cvtColor(stage_img["final"], cv2.COLOR_BGR2RGB)]
    for a, t, im in zip(ax.ravel(), titles, imgs):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t, fontsize=10); a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig1_pipeline_stages.png"), dpi=130); plt.close(fig)

    # 2. latency
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(df["frame"], df["latency_ms"], lw=0.8); ax[0].axhline(df["latency_ms"].mean(), color="r", ls="--", label="mean")
    ax[0].set(title="Per-frame processing latency", xlabel="Frame", ylabel="ms"); ax[0].legend()
    ax[1].hist(df["latency_ms"], bins=30, color="tab:blue"); ax[1].axvline(df["latency_ms"].quantile(0.95), color="r", ls="--", label="95th percentile")
    ax[1].set(title="Latency distribution", xlabel="ms", ylabel="Frames"); ax[1].legend()
    stages = ["gray", "blur", "threshold", "contours", "track"]
    means = [df[f"t_{s}"].mean() for s in stages]
    ax[2].bar(stages, means, color="tab:green"); ax[2].set(title="Mean time per stage", ylabel="ms")
    for i, v in enumerate(means):
        ax[2].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig2_latency.png"), dpi=130); plt.close(fig)

    # 3. counts over time
    fig, ax = plt.subplots(1, 2 if has_gt else 1, figsize=(14 if has_gt else 7, 4.2))
    ax = np.atleast_1d(ax)
    for c in SHAPES:
        ax[0].plot(df["frame"], df[f"n_{c}"], label=c, color=np.array(BGR[c][::-1]) / 255)
    ax[0].set(title="Detected objects per class (count vector over time)", xlabel="Frame", ylabel="Count"); ax[0].legend()
    if has_gt:
        ax[1].plot(df["frame"], df["gt_total"], label="ground truth", lw=2)
        ax[1].plot(df["frame"], df["n_total"], label="detected", lw=1)
        ax[1].set(title="Total count: detected vs ground truth", xlabel="Frame", ylabel="Objects"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig3_counts.png"), dpi=130); plt.close(fig)

    # 4. confusion matrix + ablation
    if has_gt:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        xl, yl = PRED_CLASSES + ["missed"], SHAPES + ["false alarm"]
        im = ax[0].imshow(conf, cmap="Blues")
        ax[0].set_xticks(range(len(xl))); ax[0].set_xticklabels(xl, rotation=20); ax[0].set_yticks(range(len(yl))); ax[0].set_yticklabels(yl)
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                ax[0].text(j, i, conf[i, j], ha="center", va="center", color="white" if conf[i, j] > conf.max() / 2 else "black")
        ax[0].set(title="Detection confusion matrix (all frames)", xlabel="Predicted", ylabel="Ground truth")
        names = list(ablation.keys()); f1 = [ablation[n]["macro_f1"] for n in names]; fp = [ablation[n]["fps"] for n in names]
        xs = np.arange(len(names))
        ax[1].bar(xs - 0.2, f1, 0.4, label="Macro F1", color="tab:blue")
        ax[1].set_xticks(xs); ax[1].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8); ax[1].set_ylim(0, 1.05)
        ax[1].set_ylabel("Macro F1"); ax[1].set_title("Ablation study")
        ax2 = ax[1].twinx(); ax2.bar(xs + 0.2, fp, 0.4, label="FPS", color="tab:orange"); ax2.set_ylabel("Throughput (FPS)")
        h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig4_confusion_ablation.png"), dpi=130); plt.close(fig)


# ======================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", help="'synthetic', a video file path, or a webcam index like 0")
    ap.add_argument("--out", default="cv_results")
    ap.add_argument("--frames", type=int, default=300, help="frames (synthetic video length / max frames to process)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="show a live window (press q to quit)")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--min-area", type=int, default=Params.min_area)
    ap.add_argument("--max-area", type=int, default=Params.max_area)
    args = ap.parse_args()

    P = Params(min_area=args.min_area, max_area=args.max_area)
    os.makedirs(args.out, exist_ok=True)
    figdir, framedir = os.path.join(args.out, "figures"), os.path.join(args.out, "sample_frames")
    os.makedirs(framedir, exist_ok=True)

    gt, has_gt = None, False
    if args.source == "synthetic":
        src = os.path.join(args.out, "synthetic_input.mp4")
        print("Generating synthetic test video with known ground truth ...")
        gt = make_synthetic_video(src, n_frames=args.frames, seed=args.seed)
        json.dump(gt, open(os.path.join(args.out, "synthetic_ground_truth.json"), "w"))
        has_gt = True
        cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(os.path.join(args.out, "annotated_output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))

    tracker = CentroidTracker(P.track_dist, P.track_missing)
    conf, id_state = new_confusion(), dict(map={}, switches=0)
    rows, stage_img, f, t_loop = [], None, 0, time.perf_counter()
    keep_frames = {50, 100, 150, 200}
    print(f"Processing {W}x{H} video ...")
    while f < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dets, T, dbg = process_frame(frame, P, keep_debug=(f == 100))
        t_tr = time.perf_counter()
        tracker.update(dets)
        T["track"] = (time.perf_counter() - t_tr) * 1e3
        latency = (time.perf_counter() - t0) * 1e3
        counts = {c: sum(d["cls"] == c for d in dets) for c in SHAPES}
        fps_now = 1000 / latency
        out = draw_overlay(frame, dets, tracker, counts, fps_now, latency)
        writer.write(out)

        row = dict(frame=f, latency_ms=latency, fps=fps_now, n_detections=len(dets), n_total=sum(counts.values()),
                   n_other=sum(d["cls"] == "other" for d in dets), mean_area=float(np.mean([d["area"] for d in dets])) if dets else 0.0,
                   **{f"n_{c}": counts[c] for c in SHAPES}, **{f"t_{k}": v for k, v in T.items()})
        if has_gt:
            row["gt_total"] = len(gt[f])
            for c in SHAPES:
                row[f"gt_{c}"] = sum(g["cls"] == c for g in gt[f])
            evaluate_frame(gt[f], dets, conf, id_state)
        rows.append(row)
        if f == 100:
            cimg = frame.copy()
            for d in dets:
                cv2.drawContours(cimg, [d["contour"]], -1, BGR[d["cls"]], 2)
            stage_img = dict(frame=frame, gray=dbg["gray"], smooth=dbg["smooth"], binary=dbg["binary"], contours=cimg, final=out)
        if f in keep_frames:
            cv2.imwrite(os.path.join(framedir, f"frame_{f:04d}.png"), out)
        if args.show:
            try:
                cv2.imshow("Task 4 pipeline (q to quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                args.show = False
        f += 1
    wall = time.perf_counter() - t_loop
    cap.release(); writer.release()
    if args.show:
        cv2.destroyAllWindows()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "frame_metrics.csv"), index=False)

    # ------------------------------------------------------------ summary
    lat = df["latency_ms"]
    summary = dict(source=args.source, frames=int(len(df)), resolution=f"{W}x{H}", parameters=asdict(P),
                   latency_ms=dict(mean=lat.mean(), median=lat.median(), p95=lat.quantile(0.95), max=lat.max(), min=lat.min(), std=lat.std()),
                   processing_fps=float(1000 / lat.mean()),
                   end_to_end_fps_incl_draw_and_write=float(len(df) / wall),
                   stage_mean_ms={k[2:]: float(df[k].mean()) for k in df.columns if k.startswith("t_")},
                   detections_per_frame_mean=float(df["n_detections"].mean()),
                   mean_count_vector=dict(zip(SHAPES, [float(df[f"n_{c}"].mean()) for c in SHAPES])),
                   unique_track_ids=tracker.created)
    if has_gt:
        sc = scores_from_confusion(conf)
        err = {c: float((df[f"n_{c}"] - df[f"gt_{c}"]).abs().mean()) for c in SHAPES}
        exact = float(np.mean([all(r[f"n_{c}"] == r[f"gt_{c}"] for c in SHAPES) for r in rows]))
        summary["evaluation"] = dict(per_class=sc, count_mae_per_class=err, exact_count_vector_accuracy=exact,
                                     id_switches=id_state["switches"], ground_truth_objects=len(gt[0]),
                                     confusion_matrix=conf.tolist(), confusion_rows=SHAPES + ["false alarm"], confusion_cols=PRED_CLASSES + ["missed"])
    ablation = {}
    if has_gt and not args.no_ablation:
        print("Running ablation study ...")
        no_bounds = Params(**{**asdict(P), "min_area": 1, "max_area": 10 ** 9})
        for name, (p, th, bl) in {"Full pipeline": (P, "adaptive", True), "No Gaussian blur": (P, "adaptive", False),
                                  "Global Otsu threshold": (P, "otsu", True), "No pixel bounds": (no_bounds, "adaptive", True)}.items():
            ablation[name] = run_variant(os.path.join(args.out, "synthetic_input.mp4"), gt, p, th, bl)
        pd.DataFrame(ablation).T.round(4).to_csv(os.path.join(args.out, "ablation.csv"))
        summary["ablation"] = ablation

    make_figures(figdir, rows, stage_img, conf, ablation, has_gt and bool(ablation))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)

    # ------------------------------------------------------------ console report
    print("\n================ RESULTS ================")
    print(f"Frames: {len(df)}   resolution: {W}x{H}")
    print(f"Latency per frame (ms):  mean {lat.mean():.2f} | median {lat.median():.2f} | p95 {lat.quantile(0.95):.2f} | max {lat.max():.2f}")
    print(f"Processing speed: {1000 / lat.mean():.1f} FPS   (end-to-end incl. drawing + video writing: {len(df) / wall:.1f} FPS)")
    print("Mean time per stage (ms):", {k[2:]: round(float(df[k].mean()), 3) for k in df.columns if k.startswith("t_")})
    print(f"Detections per frame: {df['n_detections'].mean():.2f}   unique track IDs created: {tracker.created}")
    if has_gt:
        e = summary["evaluation"]
        print("\nDetection quality vs ground truth:")
        for c in SHAPES:
            s = e["per_class"][c]
            print(f"  {c:10s} precision {s['precision']:.3f}  recall {s['recall']:.3f}  F1 {s['f1']:.3f}   (TP {s['tp']}, FP {s['fp']}, FN {s['fn']})")
        print(f"  Macro F1: {e['per_class']['macro_f1']:.3f}   exact count-vector accuracy: {e['exact_count_vector_accuracy']:.1%}")
        print(f"  Count MAE per class: { {k: round(v, 3) for k, v in e['count_mae_per_class'].items()} }")
        print(f"  ID switches: {e['id_switches']}  (ground-truth objects: {e['ground_truth_objects']}, track IDs created: {tracker.created})")
    if ablation:
        print("\nAblation study:")
        print(pd.DataFrame(ablation).T.round(3).to_string())
    print(f"\nAll outputs saved in '{args.out}'.")


if __name__ == "__main__":
    main()
