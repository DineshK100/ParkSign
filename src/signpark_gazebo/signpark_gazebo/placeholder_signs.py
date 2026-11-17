"""Generate synthetic placeholder sign images (signs/<class>/synthetic_*.png).

These only exist so the pipeline runs before the real photos are collected.
Real photos dropped into signs/<class>/ automatically take precedence
(see lot.find_sign_images) and are what E1 must be reported on.
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

DESIGNS = {
    'head_in': [['HEAD-IN', 'PARKING', 'ONLY'], ['PULL IN', 'PARKING'], ['FRONT-IN', 'PARKING', 'ONLY']],
    'back_in': [['BACK-IN', 'PARKING', 'ONLY'], ['REVERSE', 'PARKING', 'ONLY'], ['BACK IN', 'ONLY']],
    'parallel': [['PARALLEL', 'PARKING'], ['PARALLEL', 'PARKING', 'ONLY'], ['PARK', 'PARALLEL', 'TO CURB']],
    'angled': [['ANGLE', 'PARKING'], ['ANGLED', 'PARKING', 'ONLY'], ['ANGLE', 'PARKING', 'THIS SIDE']],
    'no_parking': [['NO', 'PARKING'], ['NO PARKING', 'ANY TIME'], ['NO', 'PARKING', 'TOW-AWAY ZONE']],
}


def draw_sign(lines, red, w=600, h=800):
    img = Image.new('RGB', (w, h), (250, 250, 250))
    d = ImageDraw.Draw(img)
    fg = (200, 20, 20) if red else (20, 90, 40)
    d.rectangle([12, 12, w - 12, h - 12], outline=fg, width=14)
    if red:  # circled P with a slash
        cx, cy, r = w // 2, 210, 130
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=22)
        d.text((cx, cy), 'P', fill=(30, 30, 30), font=ImageFont.truetype(FONT, 170), anchor='mm')
        d.line([cx - 92, cy + 92, cx + 92, cy - 92], fill=fg, width=22)
        y = 420
    else:
        y = 120
    size = 96 if max(len(s) for s in lines) <= 8 else 70
    font = ImageFont.truetype(FONT, size)
    for s in lines:
        d.text((w // 2, y), s, fill=fg, font=font, anchor='mt')
        y += int(size * 1.35)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'signs'))
    a = ap.parse_args()
    for cls, designs in DESIGNS.items():
        os.makedirs(os.path.join(a.out, cls), exist_ok=True)
        for i, lines in enumerate(designs):
            draw_sign(lines, red=(cls == 'no_parking')).save(
                os.path.join(a.out, cls, f'synthetic_{i}.png'))
    print('wrote placeholders to', os.path.abspath(a.out))


if __name__ == '__main__':
    main()
