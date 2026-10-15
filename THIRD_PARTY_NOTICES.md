# Third-party notices

ProbeGen is licensed under the Apache License, Version 2.0 (see `LICENSE`).
This file lists third-party code and data-derived material included in the
repository. Data-derived files keep the terms of their sources and are not
relicensed by the project license. It is not legal advice; follow the original
terms of each source.

## Code

`probegen_eval/depth_alignment.py` (least-squares depth alignment) is adapted
from [Marigold](https://github.com/prs-eth/Marigold) (`src/util/alignment.py`,
unused options removed), and the AbsRel metric in `probegen_eval/depth.py`
follows Marigold's `src/util/metric.py`. These sources carry the following
notice, which is retained in `depth_alignment.py`:

> Copyright 2023–2025 Marigold Team, ETH Zürich. All rights reserved.
> Licensed under the Apache License, Version 2.0.

The Apache-2.0 license text is included in `LICENSE`.

## Data-derived material

The code repository includes no benchmark images, depth maps or label images.
The manifests contain sample identifiers and relative paths; some also contain
annotations copied from the original datasets.

| Files | Source material included | Source and terms |
|---|---|---|
| `manifests/{nyuv2,diode,scannet,kitti}.json` | File paths selected from Marigold split lists | [Marigold](https://github.com/prs-eth/Marigold) `data_split/` (Apache-2.0 repository); datasets: NYU Depth V2, DIODE, ScanNet, KITTI |
| `manifests/{refcoco,refcoco+,refcocog}.json` | Referring expressions, COCO segmentation masks and annotation IDs | Expressions from RefCOCO/RefCOCO+ (UNC) and RefCOCOg (UMD) via [REFER](https://github.com/lichengunc/refer); masks from [COCO](https://cocodataset.org/#termsofuse) annotations (CC BY 4.0) |
| `manifests/ReasonSeg.json` | Text queries | [ReasonSeg / LISA](https://github.com/dvlab-research/LISA) |
| `manifests/DRAM.json` | File paths and class IDs | [DRAM](https://nadavc220.github.io/ssiap/) |
| `manifests/Manga109.json` | Page paths, category names and merged segmentation masks (RLE) | [Manga109 segmentation annotations](https://huggingface.co/datasets/MS92/MangaSegmentation), see below; pages from [Manga109](http://www.manga109.org/en/) |
| `manifests/pixmo-points.json` | Image paths, labels and counts | [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points), ODC-BY-1.0, see below |

### Manga109 segmentation annotations

The Manga109 masks are derived from the Manga Segmentation Annotations. For
each selected page, instances of one category are merged into one mask. The
annotation license requires the following credit:

> All segmentation annotations are owned and copyrighted by Minshan Xie and
> Tien-Tsin Wong.
>
> Minshan Xie, Jian Lin, Hanyuan Liu, Chengze Li, and Tien-Tsin Wong,
> "Advancing Manga Analysis: Comprehensive Segmentation Annotations for the
> Manga109 Dataset," Proceedings of the IEEE/CVF Conference on Computer Vision
> and Pattern Recognition, 2025.

The annotation JSON files also declare CC BY-NC 4.0 in their `licenses` field.
Manga109 page images are not included and require separate access from the
Manga109 project.

### PixMo-Points

PixMo-Points is licensed by Ai2 under ODC-BY-1.0 and is intended for research
and educational use in accordance with Ai2's
[Responsible Use Guidelines](https://allenai.org/responsible-use). The license
covers the annotations, not the linked images. Images remain the property of
their original hosts.

## Data archives

The separately distributed `probegen-data-*.tar.gz` archives contain copies of
images, ground-truth files and annotations from the datasets listed in
[DATA.md](DATA.md): NYU Depth V2, DIODE and KITTI (as prepared by Marigold),
MS-COCO 2014, ReasonSeg, DRAM and the PixMo-Points linked images. ScanNet and
Manga109 are not included because their terms forbid redistribution. The
archives are provided for research evaluation only. All rights remain with
the original dataset authors and image owners, and each file remains subject to
the license or terms of use of its source. Please cite the original datasets.
Rights holders who want material removed can contact
dishangzhe@sjtu.edu.cn.
