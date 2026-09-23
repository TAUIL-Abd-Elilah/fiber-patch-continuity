"""Build the blinded labeling page from a label kit (cards + order.json).

Output: <out>/index.html and <out>/cards/<id>.jpg. Only card ids and their
shuffled order go into the page; key.json (pair identities, strata, arm
decisions) never does.
"""
import argparse
import json
import os

from PIL import Image

TEMPLATE = os.path.join(os.path.dirname(__file__), 'label_page_template.html')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kit', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--kit-name', default='paris4-fiber-patch-v1')
    args = parser.parse_args()
    order = json.load(open(os.path.join(args.kit, 'order.json')))
    os.makedirs(os.path.join(args.out, 'cards'), exist_ok=True)
    kept = []
    for card_id in order:
        src = os.path.join(args.kit, 'cards', f'{card_id}.png')
        if not os.path.exists(src):
            continue
        dst = os.path.join(args.out, 'cards', f'{card_id}.jpg')
        Image.open(src).convert('RGB').save(dst, quality=86, optimize=True)
        kept.append(card_id)
    html = open(TEMPLATE, encoding='utf-8').read()
    html = html.replace('__CARDS__', json.dumps(kept)).replace('__KIT__', json.dumps(args.kit_name))
    with open(os.path.join(args.out, 'index.html'), 'w', encoding='utf-8') as handle:
        handle.write(html)
    size = sum(os.path.getsize(os.path.join(args.out, 'cards', f)) for f in os.listdir(os.path.join(args.out, 'cards')))
    print(f'{len(kept)} cards, {size / 1e6:.1f} MB of images')


if __name__ == '__main__':
    main()
