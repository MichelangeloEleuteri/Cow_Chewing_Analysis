#!/usr/bin/env python3
"""
compute_d_from_labels.py
Reads YOLO-Pose .txt labels (directory or tar.gz archives) and computes
nose-mouth Euclidean distance in pixels and normalized.
Writes CSV with columns: filename, frame, time_s, nose_x,nose_y,mouth_x,mouth_y,d_px,d_norm

Usage examples:
  python3 compute_d_from_labels.py --labels_dir /scratch/eleuteri/video3/runs_predict/video3_full_bestpt_long/labels --out /scratch/eleuteri/video3/dists_all.csv --width 1080 --height 1920 --fps 30 --plot /scratch/eleuteri/video3/dists_all.png

  python3 compute_d_from_labels.py --archive /home/eleuteri/labels_chunks_for_download/labels_chunk_000.tar.gz --out /home/eleuteri/dists_chunk_000.csv --width 1080 --height 1920 --fps 30
"""
import argparse, os, re, math, csv, tarfile
from io import TextIOWrapper

FRAME_RE = re.compile(r'(\d+)\.txt$')

def parse_label_line(line):
    toks = line.strip().split()
    if len(toks) < 5:
        return None
    try:
        vals = [float(t) for t in toks]
    except:
        # fallback: ignore malformed
        return None
    cls = int(vals[0])
    bx, by, bw, bh = vals[1:5]
    rem = vals[5:]
    kps = []
    if not rem:
        return cls, bx, by, bw, bh, kps
    # if length divisible by 3 -> triplets x y conf
    if len(rem) % 3 == 0:
        it = iter(rem)
        while True:
            try:
                x = next(it); y = next(it); c = next(it)
                kps.append((x,y,c))
            except StopIteration:
                break
    elif len(rem) % 2 == 0:
        it = iter(rem)
        while True:
            try:
                x = next(it); y = next(it)
                kps.append((x,y,None))
            except StopIteration:
                break
    else:
        # best-effort: read pairs from start
        it = iter(rem)
        while True:
            try:
                x = next(it); y = next(it)
                kps.append((x,y,None))
            except StopIteration:
                break
    return cls, bx, by, bw, bh, kps

def norm_to_px(x,y,width,height):
    return x*width, y*height

def extract_frame_index(fname):
    m = FRAME_RE.search(os.path.basename(fname))
    if m:
        return int(m.group(1))
    parts = os.path.basename(fname).rsplit('.',1)[0].split('_')
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    return None

def process_txt_handle(filename_label, fh, width, height, fps):
    rows=[]
    text = fh.read().strip().splitlines()
    for line in text:
        if not line.strip(): continue
        parsed = parse_label_line(line)
        if parsed is None: continue
        cls, bx, by, bw, bh, kps = parsed
        if len(kps) < 2:
            nose_px=(None,None); mouth_px=(None,None)
        else:
            nx,ny,_ = kps[0]
            mx,my,_ = kps[1]
            nx_px, ny_px = norm_to_px(nx, ny, width, height)
            mx_px, my_px = norm_to_px(mx, my, width, height)
            nose_px=(nx_px, ny_px); mouth_px=(mx_px, my_px)
        if nose_px[0] is None or mouth_px[0] is None:
            d_px=None; d_norm=None
        else:
            dx = nose_px[0]-mouth_px[0]
            dy = nose_px[1]-mouth_px[1]
            d_px = math.hypot(dx,dy)
            d_norm = d_px / math.hypot(width,height)
        frame = extract_frame_index(filename_label)
        time_s = (frame / fps) if (frame is not None and fps) else None
        rows.append({
            "filename": filename_label,
            "frame": frame,
            "time_s": time_s,
            "nose_x": nose_px[0],
            "nose_y": nose_px[1],
            "mouth_x": mouth_px[0],
            "mouth_y": mouth_px[1],
            "d_px": d_px,
            "d_norm": d_norm
        })
    return rows

def process_dir(labels_dir, width, height, fps):
    out=[]
    files = sorted([f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')])
    for fn in files:
        path = os.path.join(labels_dir, fn)
        try:
            with open(path,'r',encoding='utf-8') as fh:
                out.extend(process_txt_handle(fn, fh, width, height, fps))
        except Exception as e:
            # skip problematic
            continue
    return out

def process_archive(archive_path, width, height, fps):
    out=[]
    with tarfile.open(archive_path, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith('.txt')]
        for m in sorted(members, key=lambda x: x.name):
            f = tar.extractfile(m)
            if f is None: continue
            txt = TextIOWrapper(f, encoding='utf-8')
            out.extend(process_txt_handle(m.name, txt, width, height, fps))
    return out

def write_csv(rows, out_csv):
    keys=["filename","frame","time_s","nose_x","nose_y","mouth_x","mouth_y","d_px","d_norm"]
    with open(out_csv,'w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in keys})

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--labels_dir', help='directory with .txt labels')
    p.add_argument('--archive', help='one or more tar.gz archives', nargs='*')
    p.add_argument('--out', default='dists.csv')
    p.add_argument('--width', type=int, default=1080)
    p.add_argument('--height', type=int, default=1920)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--plot', help='save plot file (png)', default=None)
    args=p.parse_args()

    rows=[]
    if args.labels_dir:
        print("Processing labels dir:", args.labels_dir)
        rows = process_dir(args.labels_dir, args.width, args.height, args.fps)
    if args.archive:
        for a in args.archive:
            print("Processing archive:", a)
            rows.extend(process_archive(a, args.width, args.height, args.fps))

    print("Total entries:", len(rows))
    write_csv(rows, args.out)
    print("Wrote:", args.out)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            times=[r['time_s'] if r['time_s'] is not None else r['frame'] for r in rows if r['d_px'] is not None]
            dvals=[r['d_px'] for r in rows if r['d_px'] is not None]
            plt.figure(figsize=(12,4))
            plt.plot(times, dvals, linewidth=0.5, marker='.', markersize=2)
            plt.xlabel('time (s) or frame'); plt.ylabel('distance (px)')
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print("Saved plot:", args.plot)
        except Exception as e:
            print("Plotting skipped:", e)

if __name__=='__main__':
    main()
