"""E1: offline sign-classification evaluation (VLM vs. OCR).

Images are labeled by folder name:  <dir>/<class>/<anything>.{jpg,png}
Crops saved by the sign reader in sim are labeled by file name: <spot>_<class>_<n>.png

    ros2 run signpark_perception eval_signs --dir src/signpark_gazebo/signs --backends ollama ocr
    ros2 run signpark_perception eval_signs --crops ~/.ros/signpark/scenario_1/crops --backends ollama

Writes a CSV of every prediction and prints accuracy + confusion matrix per backend.
"""
import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image

from signpark_perception.classifiers import SIGN_CLASSES, make_classifier

LABELS = SIGN_CLASSES + ['unknown']


def collect(args):
    items = []
    exts = ('.jpg', '.jpeg', '.png')
    if args.dir:
        for cls in SIGN_CLASSES:
            for f in sorted(glob.glob(os.path.join(args.dir, cls, '*'))):
                if f.lower().endswith(exts) and (args.include_synthetic or not os.path.basename(f).startswith('synthetic_')):
                    items.append((f, cls))
    if args.crops:
        for f in sorted(glob.glob(os.path.join(args.crops, '*.png'))):
            parts = os.path.basename(f)[:-4].split('_')
            cls = '_'.join(parts[1:-1])
            if cls in SIGN_CLASSES:
                items.append((f, cls))
    return items


def confusion(rows, backend):
    m = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for r in rows:
        if r['backend'] == backend:
            m[LABELS.index(r['truth']), LABELS.index(r['pred'])] += 1
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', help='folder with <class>/ sub-folders of photos')
    ap.add_argument('--crops', help='folder of sim crops saved by sign_reader')
    ap.add_argument('--backends', nargs='+', default=['ollama'], help='ollama | claude | ocr | ground_truth')
    ap.add_argument('--model', default='', help='model name for ollama/claude (default: qwen2.5vl:3b / claude-opus-5-5)')
    ap.add_argument('--include-synthetic', action='store_true', help='also evaluate synthetic_* placeholders')
    ap.add_argument('--out', default='results/e1_signs.csv')
    args = ap.parse_args()

    items = collect(args)
    if not items:
        ap.error('no labeled images found (real photos go in signs/<class>/; use --include-synthetic to test with placeholders)')
    print(f'{len(items)} images')
    rows = []
    for b in args.backends:
        clf = make_classifier(b, **({'model': args.model} if args.model and b in ('ollama', 'claude') else {}))
        for f, truth in items:
            img = np.array(Image.open(f).convert('RGB'))
            r = clf.classify(img, truth) if b == 'ground_truth' else clf.classify(img)
            rows.append(dict(backend=b, file=f, truth=truth, pred=r.sign_class, text=r.text_read,
                             confidence=r.confidence, latency_s=round(r.latency_s, 3)))
            print(f'  [{b}] {"OK " if r.sign_class == truth else "ERR"} {os.path.basename(f)}: {r.sign_class} (truth {truth})')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for b in args.backends:
        rb = [r for r in rows if r['backend'] == b]
        acc = np.mean([r['pred'] == r['truth'] for r in rb])
        lat = np.mean([r['latency_s'] for r in rb])
        print(f'\n=== {b}: accuracy {acc * 100:.1f}% ({len(rb)} images), mean latency {lat:.2f}s')
        for cls in SIGN_CLASSES:
            rc = [r for r in rb if r['truth'] == cls]
            if rc:
                print(f'  {cls:<11} {np.mean([r["pred"] == cls for r in rc]) * 100:5.1f}%  (n={len(rc)})')
        m = confusion(rows, b)
        print('  confusion (rows = truth, cols = prediction):')
        print('  ' + ' ' * 11 + ' '.join(f'{lab[:6]:>6}' for lab in LABELS))
        for i, lab in enumerate(LABELS[:-1]):
            print(f'  {lab:<11}' + ' '.join(f'{v:>6}' for v in m[i]))
    print(f'\nper-image results: {args.out}')


if __name__ == '__main__':
    main()
