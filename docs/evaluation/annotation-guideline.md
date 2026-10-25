# Annotation guideline — evaluation set

The evaluation set is the ground truth every model decision is measured against (#19).
Consistency matters more than speed: two people following this guide should draw the same boxes.

## What to label

Each image gets **one box per visible product unit** (each *facing*): a bottle, can, pack, box or bag
that a shopper could pick up.

| Label it | Do not label it |
|---|---|
| Every product unit whose type you can recognise | Price tags, shelf edges, shelf labels |
| Units partly hidden, if **about half or more** of the front is visible | Units hidden more than about half, or too blurry to recognise |
| Units on every shelf, including top and bottom | Posters, signs, people, carts, empty shelf space |
| Units seen from the side if they are clearly products | Products reflected in glass or mirrors |

**Stacks and rows:** label each unit you can see as its own box. Units hidden behind the front
unit are not labelled.

## How to draw the box

- **Tight** around the visible part of the unit: no extra background, nothing cut off.
- Partly hidden unit: box only the **visible** part.
- Do not merge neighbouring units into one box, even if they are identical.
- Pre-filled boxes come from the current detector: **correct** them (move, resize), **delete**
  wrong ones (price tags, merged units, background) and **add** missed ones. Never keep a box
  just because it was pre-filled.

## Groups (same product)

Every box gets a short **group code**, e.g. `a`, `b`, `c` or `1`, `2`, `3`.

- Same code = **the same product**: same brand, variant/flavour and size.
- Different flavour, colour variant or size = **different code**.
- Codes only need to be consistent **within one image**; `a` in image 1 and `a` in image 2 are unrelated.
- If two units look identical but you are not sure, give them the **same** code.
- A unit unlike any other gets its own code.

Tip: label all units of one product, then move on to the next product. Keeping a short note
per image (`a = red can, b = blue can`) helps.

## Quality bar

- Aim for every product in the image boxed and grouped; unsure cases follow the rules above.
- Expect about 15–30 minutes per image depending on shelf density.
- When an image is done, zoom out and check: no unboxed products, no box without a group code.

## Files

| File | What |
|---|---|
| `evaluation/labelstudio_config.xml` | Label Studio interface (boxes + per-box group code) |
| `scripts/eval/prelabel.py` | Writes `data/eval/prelabels.json`: Label Studio tasks pre-filled with detector boxes |
| `scripts/eval/convert_labelstudio.py` | Label Studio JSON export -> `data/eval/annotations.json` (with image checksums) |
| `data/eval/annotations.json` | **The evaluation set** (committed). Images are referenced from `sample_images/` |

Evaluation images are **never** used for training or for tuning by eye.

See [labelling-setup.md](labelling-setup.md) for the step-by-step tool setup.
