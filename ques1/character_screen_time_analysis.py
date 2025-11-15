"""
Character Screen-Time & Clip Extraction Pipeline
Author: Utkarsh Suryaman
Date: 2025-11-15

What this script does (end-to-end):
1. Loads a local video file (downloaded from YouTube by you).
2. Extracts audio and runs Whisper (openai/whisper) for timestamps & transcripts.
3. Samples frames and detects faces (face_recognition/OpenCV).
4. Computes face embeddings and clusters them to find distinct characters.
5. Builds continuous occurrence segments (start/end times) per cluster.
6. Picks top-5 most-visible characters by total screen time.
7. For each top character: creates an Excel sheet listing all occurrences and overall screentime.
8. Selects top-3 recommended clips (>=5s) per character using heuristics (longest continuous presence, centrality, face-size), generates titles and short reasoning.
9. Matches transcript lines to on-screen times and ranks "punchy" dialogues using heuristics (length, sentiment, TF-IDF, punctuation). Exports top-5 punchy lines with reasoning.

Notes & requirements:
- This script prints detailed logs so an interviewer can follow progress.
- Libraries required (suggested install commands):
    pip install opencv-python-headless numpy pandas openpyxl moviepy face_recognition sklearn tqdm librosa pydub transformers "whisper" webrtcvad
  * face_recognition may require dlib & C++ build tools. If that's hard, use facenet-pytorch + MTCNN + InceptionResnetV1 instead (I can provide that variant).
- Whisper model: small/medium recommended depending on CPU/GPU. Adjust model name in config.
- This is a practical, heuristic pipeline — tune thresholds for your specific clip.

Run example:
    python character_screen_time_analysis.py --video path/to/video.mp4 --whisper_model small --fps 1 --min_clip_sec 5

"""
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
import os
import sys
import argparse
import tempfile
import math
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Face libs
import face_recognition
from sklearn.cluster import DBSCAN

# Audio / ASR
import whisper

from moviepy.editor import VideoFileClip

# For simple scoring
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Utility functions ----------

def print_banner(msg):
    print("\n" + "="*80)
    print(msg)
    print("="*80 + "\n")


def extract_audio(video_path, out_audio_path):
    print(f"Extracting audio from {video_path} -> {out_audio_path}")
    clip = VideoFileClip(str(video_path))
    clip.audio.write_audiofile(str(out_audio_path), verbose=False, logger=None)
    clip.reader.close()
    clip.audio.reader.close_proc()


def sample_frames(video_path, fps_sampling=1.0, save_frames=False, frame_out_dir=None):
    """Sample frames at fps_sampling (frames per second). Returns list of tuples (t_sec, frame_bgr).
    """
    print(f"Sampling frames from {video_path} at {fps_sampling} FPS")
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / video_fps if video_fps else 0
    print(f"Video FPS={video_fps:.2f}, total_frames={total_frames}, approx_duration={duration:.2f}s")

    frames = []
    times = np.arange(0, duration, 1.0 / fps_sampling)
    for t in tqdm(times, desc="Sampling frames"):
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append((t, frame.copy()))
        if save_frames and frame_out_dir is not None:
            fname = Path(frame_out_dir) / f"frame_{int(t*1000)}ms.jpg"
            cv2.imwrite(str(fname), frame)
    cap.release()
    print(f"Sampled {len(frames)} frames")
    return frames


def detect_faces_in_frames_mtcnn(frames, resize_to=None, min_face_size=20):
    """
    Detect faces using facenet-pytorch MTCNN and compute embeddings with InceptionResnetV1.
    Returns list of dicts: { time, boxes, crops, embeddings }
    - boxes: list of (left, top, right, bottom)
    - crops: list of numpy arrays (bgr uint8) useful for debug/save
    - embeddings: list of 512-d numpy arrays (float32) or torch tensors on cpu
    Prints progress and counts.
    """
    print("Detecting faces with MTCNN and computing embeddings with InceptionResnetV1")
    results = []
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device for face models: {device_str}")
    for t, frame in tqdm(frames, desc="MTCNN detect+embed"):
        # mtcnn expects PIL or ndarray in RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # mtcnn( ) with keep_all=True returns cropped PIL images when `return_prob=False`
        # Use mtcnn.detect to get boxes, then mtcnn.extract? Simpler: call mtcnn on the frame to get boxes and face tensors.
        # We'll use mtcnn to return cropped face tensors.
        try:
            # faces = list of PIL.Image or torch.Tensor depending on mtcnn config
            # boxes: numpy array Nx4 in (x1,y1,x2,y2) or None
            boxes, probs = mtcnn.detect(rgb)
            crops = []
            embeddings = []
            boxes_list = []
            if boxes is None:
                results.append({'time': t, 'boxes': [], 'crops': [], 'embeddings': []})
                continue

            # mtcnn.extract one-by-one to get aligned faces: use mtcnn.forward? easiest: use mtcnn to crop using .crop
            # facenet-pytorch's MTCNN can be used to return face tensors via mtcnn(rgb, return_prob=True) but that returns stacked tensors.
            faces_tensors = mtcnn(rgb, return_prob=False)  # returns None or list of tensors or a single tensor
            # faces_tensors can be a torch.Tensor (N,3,160,160) or a single tensor when N==1
            if faces_tensors is None:
                results.append({'time': t, 'boxes': [], 'crops': [], 'embeddings': []})
                continue

            # normalize shapes: make list of face tensors
            if isinstance(faces_tensors, torch.Tensor):
                face_t_list = [faces_tensors[i].to(device) for i in range(faces_tensors.shape[0])]
            else:
                face_t_list = [f.to(device) for f in faces_tensors]

            # convert boxes to int list
            for b in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                if w < min_face_size or h < min_face_size:
                    continue
                boxes_list.append({'left': x1, 'top': y1, 'right': x2, 'bottom': y2, 'w': w, 'h': h,
                                   'cx': x1 + w/2, 'cy': y1 + h/2})
                # extract crop for optional saving/debug
                crop = frame[y1:y2, x1:x2].copy() if y2>y1 and x2>x1 else np.zeros((min_face_size,min_face_size,3), dtype=np.uint8)
                crops.append(crop)

            # compute embeddings in batch for all face_t_list
            if len(face_t_list) > 0:
                # create batch
                batch = torch.stack(face_t_list).to(device)
                with torch.no_grad():
                    embs = resnet(batch).cpu().numpy()  # shape (N,512)
                embeddings = [emb.astype(np.float32) for emb in embs]
            else:
                embeddings = []

            # ensure lengths align: boxes_list and embeddings should correspond in order returned by mtcnn
            # NOTE: MTCNN returns detections in same order as faces_tensors
            results.append({'time': t, 'boxes': boxes_list, 'crops': crops, 'embeddings': embeddings})
        except Exception as e:
            print(f"Warning: MTCNN error at t={t:.2f}s: {e}")
            results.append({'time': t, 'boxes': [], 'crops': [], 'embeddings': []})
    # summary
    total_faces = sum([len(r['embeddings']) for r in results])
    print(f"MTCNN detected {total_faces} face crops across {len(results)} sampled frames")
    return results


def cluster_face_encodings(all_encodings, eps=0.5, min_samples=3):
    if len(all_encodings) == 0:
        return np.array([])
    X = np.stack(all_encodings)
    print(f"Clustering {len(X)} face encodings with DBSCAN (eps={eps}, min_samples={min_samples})")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X)
    labels = clustering.labels_
    return labels


def build_occurrences(detections, labels_per_encoding, frame_times, sampling_interval):
    """Aggregate per-cluster continuous occurrences. detections: per-frame dicts. labels_per_encoding maps each detected encoding (in order seen) to label.
    Returns dict: cluster_id -> list of (start_sec, end_sec, frames_count, avg_box)
    """
    cluster_occ = defaultdict(list)
    # We'll iterate frame by frame and see which clusters are present
    current = {}  # cluster_id -> start_time and accum stats
    for i, det in enumerate(detections):
        t = det['time']
        # which clusters present in this frame
        clusters_here = []
        # encodings correspond to a global order; labels_per_encoding used externally. Instead, we'll expect that the caller attaches labels to each encoding in detections.
        if 'labels' in det:
            for j, lbl in enumerate(det['labels']):
                if lbl == -1:
                    continue
                clusters_here.append((lbl, det['boxes'][j]))
        present_clusters = set([c for c,_ in clusters_here])
        # handle existing
        for c in list(current.keys()):
            if c in present_clusters:
                # extend
                current[c]['end'] = t
                current[c]['frames'] += 1
                current[c]['boxes'].append([b['left'], b['top'], b['right'], b['bottom'], b['w'], b['h']]) if c in present_clusters else None
            else:
                # close
                cluster_occ[c].append((current[c]['start'], current[c]['end'], current[c]['frames'], current[c].get('boxes', [])))
                del current[c]
        # open new
        for (c, b) in clusters_here:
            if c not in current:
                current[c] = {'start': t, 'end': t, 'frames': 1, 'boxes': [[b['left'], b['top'], b['right'], b['bottom'], b['w'], b['h']]]}
    # close any remaining
    for c in list(current.keys()):
        cluster_occ[c].append((current[c]['start'], current[c]['end'], current[c]['frames'], current[c].get('boxes', [])))
    return cluster_occ


def merge_close_segments(segments, max_gap=1.0):
    """Merge segments that are separated by <= max_gap seconds."""
    if not segments:
        return []
    segments_sorted = sorted(segments, key=lambda x: x[0])
    merged = []
    cur_s, cur_e, *rest = list(segments_sorted[0])
    cur_frames = segments_sorted[0][2]
    cur_boxes = segments_sorted[0][3] if len(segments_sorted[0])>3 else []
    for seg in segments_sorted[1:]:
        s, e, frames, boxes = seg
        if s - cur_e <= max_gap:
            cur_e = e
            cur_frames += frames
            cur_boxes += boxes
        else:
            merged.append((cur_s, cur_e, cur_frames, cur_boxes))
            cur_s, cur_e, cur_frames, cur_boxes = s, e, frames, boxes
    merged.append((cur_s, cur_e, cur_frames, cur_boxes))
    return merged


def select_top_clips(segments, min_duration=5.0, top_k=3):
    """Choose top_k clips (>=min_duration) from segments using heuristics. Returns list of (start,end,duration,score)
    Heuristic score: duration primarily, then avg box area if available.
    """
    scored = []
    for (s,e,frames,boxes) in segments:
        dur = e - s
        if dur < min_duration:
            continue
        area_score = 0.0
        if boxes:
            boxes_np = np.array(boxes)
            areas = boxes_np[:,4] * boxes_np[:,5]
            area_score = float(np.mean(areas))
        score = dur + 0.0001 * area_score
        scored.append((s,e,dur,score))
    scored_sorted = sorted(scored, key=lambda x: x[3], reverse=True)
    return scored_sorted[:top_k]


def run_whisper_transcription(audio_path, model_name='small'):
    print(f"Loading Whisper model '{model_name}' (this may take a while)")
    model = whisper.load_model(model_name)
    print(f"Transcribing audio: {audio_path}")
    result = model.transcribe(str(audio_path), task='transcribe')
    # result contains 'segments' with start/end/text
    segments = result.get('segments', [])
    print(f"Whisper produced {len(segments)} transcript segments")
    return segments


def rank_punchy_dialogues(segments_text, top_k=5):
    """Heuristic ranking: prefer medium-length lines, containing punctuation (!) or strong words, TF-IDF importance, and brevity.
    segments_text: list of dicts with 'start','end','text'
    Returns top_k lines with a short reason.
    """
    texts = [s['text'].strip() for s in segments_text]
    if len(texts) == 0:
        return []
    vec = TfidfVectorizer(stop_words='english').fit_transform(texts)
    scores = vec.sum(axis=1).A1  # naive importance
    # punctuation bonus
    punct_bonus = np.array([1.5 if ('!' in t or '?' in t or '...' in t) else 1.0 for t in texts])
    length_penalty = np.array([1.0 / (1.0 + abs(len(t.split()) - 6)) for t in texts])  # prefer ~6-word lines

    combined = scores * punct_bonus * length_penalty
    idxs = np.argsort(-combined)[:top_k]
    results = []
    for i in idxs:
        results.append({'start': segments_text[i]['start'], 'end': segments_text[i]['end'], 'text': texts[i], 'score': float(combined[i]), 'reason': 'High TF-IDF importance and punchy punctuation/length.'})
    return results


# ---------- Main pipeline ----------

def main(args):
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) extract audio for Whisper
    audio_path = out_dir / "extracted_audio.wav"
    extract_audio(video_path, audio_path)

    # 2) run Whisper transcription
    transcript_segments = run_whisper_transcription(audio_path, model_name=args.whisper_model)

    # 3) sample frames
    frames = sample_frames(video_path, fps_sampling=args.fps)

    # 4) detect faces and encodings
    detections = detect_faces_in_frames_mtcnn(frames)

    # flatten encodings and keep mapping
    # --- Flatten embeddings (adapted for MTCNN outputs) ---
    all_encodings = []
    mapping = []  # tuple(frame_idx, encoding_idx_in_frame)
    for i, det in enumerate(detections):
        for j, emb in enumerate(det.get('embeddings', [])):
            all_encodings.append(emb)           # emb is a 512-d numpy array
            mapping.append((i, j))
    print(f"Total face detections (embeddings): {len(all_encodings)}")


    # 5) cluster encodings
    if len(all_encodings) == 0:
        print("No faces found. Exiting.")
        return
    labels = cluster_face_encodings(all_encodings, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)

    # attach labels back to detections
    for (label, (frame_i, enc_j)) in zip(labels, mapping):
     detections[frame_i].setdefault('labels', []).append(int(label))

    # Note: frames with no encodings won't have 'labels'

    # 6) build occurrences per cluster
    occurrences = build_occurrences(detections, labels, [f[0] for f in frames], sampling_interval=1.0/args.fps)

    # merge close segments
    merged_occ = {c: merge_close_segments(occs, max_gap=args.merge_gap) for c,occs in occurrences.items()}

    # compute total screen time per cluster
    cluster_times = {}
    for c, segs in merged_occ.items():
        total_sec = sum([(e-s) for (s,e,_,__) in segs])
        cluster_times[c] = total_sec
    # find top 5 clusters by time
    sorted_clusters = sorted(cluster_times.items(), key=lambda x: x[1], reverse=True)
    print_banner("Screen time per cluster (all clusters)")
    for c,sec in sorted_clusters:
        print(f"Cluster {c}: {sec:.2f}s (segments: {len(merged_occ.get(c,[]))})")

    top_clusters = [c for c,_ in sorted_clusters[:args.top_k]]
    print(f"Top {args.top_k} clusters: {top_clusters}")

    # Prepare Excel writer
    xls_path = out_dir / "character_screen_time_report.xlsx"
    writer = pd.ExcelWriter(str(xls_path), engine='openpyxl')

    # For dialogue matching: use whisper segments which have start/end/text
    transcript_for_matching = [{'start': s['start'], 'end': s['end'], 'text': s['text']} for s in transcript_segments]

    # 7) For each top cluster, export occurrences and select top clips
    report_summary = []
    for rank, c in enumerate(top_clusters, start=1):
        segs = merged_occ.get(c, [])
        total_sec = cluster_times.get(c, 0.0)
        print_banner(f"Processing Cluster {c} (rank {rank}) total_screen_time={total_sec:.2f}s segments={len(segs)}")
        # Build a sheet DataFrame for occurrences
        rows = []
        for (s,e,frames,boxes) in segs:
            rows.append({'start_sec': s, 'end_sec': e, 'duration_sec': e-s, 'frames': frames})
        df_occ = pd.DataFrame(rows).sort_values('start_sec')
        if df_occ.empty:
            df_occ = pd.DataFrame(columns=['start_sec','end_sec','duration_sec','frames'])
        df_occ.to_excel(writer, sheet_name=f'Character_{c}_occ', index=False)
        print(f"Wrote sheet: Character_{c}_occ with {len(df_occ)} rows")

        # Select top clips
        top_clips = select_top_clips(segs, min_duration=args.min_clip_sec, top_k=args.clips_per_char)
        clips_info = []
        for i,(s,e,dur,score) in enumerate(top_clips):
            title = f"TopClip_C{c}_{i+1}_{int(s)}s"  # simple generated title
            # reasoning: longer presence + central face size
            reason = f"Chosen because character is continuously on-screen for {dur:.1f}s. Score={score:.2f}."
            clips_info.append({'rank': i+1, 'start': s, 'end': e, 'duration': dur, 'title': title, 'reason': reason})
            print(f"Selected clip #{i+1} for C{c}: {s:.1f}-{e:.1f}s ({dur:.1f}s) -> {title}")
        df_clips = pd.DataFrame(clips_info)
        if df_clips.empty:
            df_clips = pd.DataFrame(columns=['rank','start','end','duration','title','reason'])
        df_clips.to_excel(writer, sheet_name=f'Character_{c}_clips', index=False)
        print(f"Wrote sheet: Character_{c}_clips with {len(df_clips)} rows")

        # Find associated transcript lines overlapping each clip
        dialog_rows = []
        for clip in clips_info:
            s,e = clip['start'], clip['end']
            overlapping = [t for t in transcript_for_matching if not (t['end'] < s or t['start'] > e)]
            for t in overlapping:
                dialog_rows.append({'clip_title': clip['title'], 'text': t['text'].strip(), 'start': t['start'], 'end': t['end']})
        df_dialogs = pd.DataFrame(dialog_rows)
        if df_dialogs.empty:
            df_dialogs = pd.DataFrame(columns=['clip_title','text','start','end'])
        df_dialogs.to_excel(writer, sheet_name=f'Character_{c}_dialogs', index=False)

        report_summary.append({'cluster': c, 'total_screen_time_s': total_sec, 'num_segments': len(segs)})

    # 8) Global top-5 punchy dialogues
    punchy = rank_punchy_dialogues(transcript_for_matching, top_k=args.punchy_k)
    df_punch = pd.DataFrame(punchy)
    if df_punch.empty:
        df_punch = pd.DataFrame(columns=['start','end','text','score','reason'])
    df_punch.to_excel(writer, sheet_name='Top_Punchy_Dialogues', index=False)
    print(f"Wrote sheet: Top_Punchy_Dialogues with {len(df_punch)} rows")

    # Write summary sheet
    df_summary = pd.DataFrame(report_summary)
    df_summary.to_excel(writer, sheet_name='Summary', index=False)
    writer.close()
    print_banner(f"Report saved to: {xls_path}")

    # Optionally: write out the recommended clip video files (requires ffmpeg via moviepy)
    if args.export_clips:
        print("Exporting recommended clips as separate mp4 files...")
        clip_writer = VideoFileClip(str(video_path))
        for rank, c in enumerate(top_clusters, start=1):
            segs = merged_occ.get(c, [])
            top_clips = select_top_clips(segs, min_duration=args.min_clip_sec, top_k=args.clips_per_char)
            for i,(s,e,dur,score) in enumerate(top_clips):
                out_clip_path = out_dir / f"Character{c}_TopClip_{i+1}_{int(s)}_{int(e)}.mp4"
                print(f"Writing clip file: {out_clip_path} ({s:.1f}-{e:.1f}s)")
                sub = clip_writer.subclip(s, e)
                sub.write_videofile(str(out_clip_path), codec='libx264', audio_codec='aac', verbose=False, logger=None)
        clip_writer.reader.close()
        clip_writer.audio.reader.close_proc()

    print_banner("Pipeline completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Character screen-time & clip extraction pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to local video file')
    parser.add_argument('--out_dir', type=str, default='output_report', help='Output folder')
    parser.add_argument('--whisper_model', type=str, default='small', help='Whisper model to use (tiny, small, medium, large)')
    parser.add_argument('--fps', type=float, default=1.0, help='Frame sampling rate (frames/sec)')
    parser.add_argument('--face_model', type=str, default='hog', help='face_recognition model: hog or cnn')
    parser.add_argument('--dbscan_eps', type=float, default=0.5, help='DBSCAN eps for face clustering')
    parser.add_argument('--dbscan_min_samples', type=int, default=3, help='DBSCAN min_samples')
    parser.add_argument('--merge_gap', type=float, default=1.0, help='Max gap to merge occurrences (sec)')
    parser.add_argument('--top_k', type=int, default=5, help='How many top characters to pick')
    parser.add_argument('--min_clip_sec', type=float, default=5.0, help='Minimum clip length to consider (sec)')
    parser.add_argument('--clips_per_char', type=int, default=3, help='How many clips to recommend per character')
    parser.add_argument('--punchy_k', type=int, default=5, help='How many punchy dialogues to select')
    parser.add_argument('--export_clips', action='store_true', help='Whether to export clip video files')
    args = parser.parse_args()

    main(args)
