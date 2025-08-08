import librosa
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict

def frames_from_segment(start, end, sr, hop_length=512, frame_length=2048):
    # Not used; we will extract MFCC features per short window in segment.
    pass

def diarize_file(audio_path: str, segments: List[Dict], sr=16000):
    """
    Very simple diarizer for up to 2 speakers:
    - Break audio into short windows (e.g., 1s or segment granularity)
    - Extract MFCCs and average per window
    - KMeans(k=2) to cluster to two speakers
    - Map segments timestamps to nearest cluster label
    """
    y, sr_real = librosa.load(audio_path, sr=sr)
    # Build sampling windows of 0.8s with hop 0.4s
    win = 0.8
    hop = 0.4
    wlen = int(win * sr)
    hlen = int(hop * sr)
    feats = []
    times = []
    for start in range(0, max(1, len(y)-wlen), hlen):
        frame = y[start:start+wlen]
        if len(frame) < wlen: break
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        feats.append(mfcc_mean)
        times.append((start/sr, (start+wlen)/sr))
    feats = np.array(feats)
    if len(feats) < 2:
        # fallback: single speaker
        diarized_text = {"speaker_1": []}
        for s in segments:
            diarized_text["speaker_1"].append({"start": s['start'], "end": s['end'], "text": s['text']})
        return diarized_text

    kmeans = KMeans(n_clusters=2, random_state=0).fit(feats)
    labels = kmeans.labels_

    # For each whisper segment, pick the window index whose midpoint is closest to segment midpoint, then assign its label.
    diarized = {"speaker_1": [], "speaker_2": []}
    for s in segments:
        mid = (s['start'] + s['end'])/2.0
        # find closest window
        diffs = [abs((t0+t1)/2.0 - mid) for (t0,t1) in times]
        idx = int(np.argmin(diffs))
        label = labels[idx]
        spk = f"speaker_{label+1}"
        diarized[spk].append({"start": s['start'], "end": s['end'], "text": s['text']})
    # Merge contiguous segments per speaker into readable text if desired
    merged = {}
    for k,v in diarized.items():
        merged[k] = _merge_segments(v)
    return merged

def _merge_segments(seglist):
    if not seglist:
        return []
    out = []
    cur = seglist[0].copy()
    for s in seglist[1:]:
        if abs(s['start'] - cur['end']) < 0.6:  # join if close
            cur['end'] = s['end']
            cur['text'] = cur['text'] + " " + s['text']
        else:
            out.append(cur)
            cur = s.copy()
    out.append(cur)
    return out
