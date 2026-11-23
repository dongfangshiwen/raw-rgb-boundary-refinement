"""Inventory native 19-class Foggy Driving labels before new model evaluation."""
import re

import numpy as np
from PIL import Image

from scripts.data_alignment_v15 import rgb_signature
from scripts.projective_ledger_v12 import REPO, digest, now, read, write


DATA = REPO.parent / 'benchmark_data/foggy-driving/Foggy_Driving'
OUT = REPO / 'results/real-fog-alignment-v17'
PARENT = REPO / 'protocols/data-alignment-v15/manifest.json'
# The official Cityscapes trainId order. Source mask values are labelIds.
LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
NAMES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
         'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')


def city_train_ids(labels):
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError('Original 8-bit single-channel labelIds required')
    lut = np.full(256, 255, np.uint8)
    lut[list(LABEL_IDS)] = np.arange(19, dtype=np.uint8)
    mapped = lut[labels]
    if int((mapped != 255).sum()) != int(np.isin(labels, LABEL_IDS).sum()):
        raise ValueError('Unexpected valid pixel count')
    return mapped


def load_list(kind):
    path = DATA / 'lists_file_names' / f'leftImg8bit_test{kind}_filenames.txt'
    rows = path.read_text(encoding='utf-8-sig').splitlines()
    if len(set(rows)) != len(rows) or any(not (DATA / row).is_file() for row in rows):
        raise ValueError('Missing or repeated official subset entry')
    return set(rows), digest(path)


def main():
    parent = read(PARENT)
    all_entries, all_sha = load_list('all')
    fine, fine_sha = load_list('fine')
    dense, dense_sha = load_list('dense')
    if not (fine <= all_entries and dense <= all_entries and (len(all_entries), len(fine), len(dense)) == (101, 33, 21)):
        raise ValueError('Official image subsets changed')
    if all_entries != {p.relative_to(DATA).as_posix() for p in DATA.rglob('*_leftImg8bit.png')}:
        raise ValueError('Official all list differs from files')
    source_hashes = {r['decoded_RGB_SHA256'] for r in parent['semantic_records']}
    for row in parent['RGB_pairs']:
        source_hashes.update((row['hazy']['decoded_RGB_SHA256'], row['clear']['decoded_RGB_SHA256']))
    rows = []
    for name in sorted(all_entries):
        path = DATA / name
        ident = path.stem.removesuffix('_leftImg8bit')
        kind = 'gtFine' if name in fine else 'gtCoarse'
        other = 'gtCoarse' if kind == 'gtFine' else 'gtFine'
        annotation = DATA / path.relative_to(DATA).as_posix().replace('leftImg8bit/', kind + '/', 1)
        annotation = annotation.with_name(ident + '_' + kind + '_labelIds.png')
        other_label = DATA / annotation.relative_to(DATA).as_posix().replace(kind, other)
        if not annotation.is_file() or other_label.exists() or not re.fullmatch(r'.+_.+_.+', ident):
            raise ValueError('Ambiguous original match: ' + name)
        with Image.open(path) as im:
            im.load()
            size = im.size
            decoded = rgb_signature(im)
        with Image.open(annotation) as im:
            target = np.asarray(im)
            if im.size != size or np.any((target > 33) & (target != 255)):
                raise ValueError('Unexpected source label IDs or geometry: ' + name)
            mapped = city_train_ids(target)
            counts = np.bincount(mapped.reshape(-1), minlength=256)
        rows.append(dict(id=ident, rgb=str(path), annotation=str(annotation), size=list(size),
                         annotation_kind=kind, dense=name in dense, rgb_SHA256=digest(path),
                         mask_SHA256=digest(annotation), decoded_RGB_SHA256=decoded,
                         exact_overlap_with_V15_sources=decoded in source_hashes,
                         counts_19=counts[:19].tolist(), void_pixels=int(counts[255])))
    if len({r['id'] for r in rows}) != 101:
        raise ValueError('Repeated image IDs')
    seen = {r['decoded_RGB_SHA256'] for r in rows}
    scopes = {'all': rows, 'fine': [r for r in rows if r['annotation_kind'] == 'gtFine'],
              'coarse': [r for r in rows if r['annotation_kind'] == 'gtCoarse'],
              'dense': [r for r in rows if r['dense']]}
    counts = {k: dict(images=len(values), class_pixels=np.sum([r['counts_19'] for r in values], axis=0).tolist(),
                      void_pixels=sum(r['void_pixels'] for r in values),
                      total_pixels=sum(r['size'][0] * r['size'][1] for r in values)) for k, values in scopes.items()}
    if any(sum(v['class_pixels']) + v['void_pixels'] != v['total_pixels'] for v in counts.values()):
        raise ValueError('Native pixel accounting failed')
    result = dict(all_passed=True, data_root=str(DATA), original_GT_unchanged=True,
                  source_url='https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/',
                  label_url='https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py',
                  official_lists_SHA256=dict(all=all_sha, fine=fine_sha, dense=dense_sha),
                  parent_manifest_SHA256=digest(PARENT), label_id_to_train_id=dict(zip(LABEL_IDS, range(19))),
                  class_names=NAMES, ignored_train_id=255, subsets=counts,
                  dense_and_fine=len(dense & fine), exact_RGB_overlaps_with_V15=sum(r['exact_overlap_with_V15_sources'] for r in rows),
                  exact_repeated_decoded_RGB_in_dataset=101 - len(seen), rows=rows,
                  scope='Real fog, semantics-only, evaluation-only, previously examined in binary experiments. Native Cityscapes19 ontology; no CamVid11 or crosswalk remapping. No paired clear RGB or claim of location independence.',
                  new_test_predictions=0, training_updates=0, finished_utc=now())
    write(OUT / 'manifest.json', result)
    write(OUT / 'summary.json', {k: v for k, v in result.items() if k != 'rows'})
    print(dict(images=101, fine=33, coarse=68, dense=21, dense_and_fine=len(dense & fine),
               exact_overlap=result['exact_RGB_overlaps_with_V15'], distinct_decoded=len(seen),
               supported_classes={k: sum(n > 0 for n in v['class_pixels']) for k, v in counts.items()},
               new_test_predictions=0), flush=True)


if __name__ == '__main__':
    main()
