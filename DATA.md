# Data

ProbeGen evaluates 500 samples per benchmark, listed in `manifests/<dataset>.json`
(input image, query, ground truth and prediction filename). RefCOCO/RefCOCO+/RefCOCOg
and Manga109 masks are stored in the manifests, so only their images are needed.

## Download

Nine benchmarks are available in seven archives from [Hugging Face](https://huggingface.co/datasets/Becomebright/ProbeGen):

| Archive | Benchmark(s) | Size |
|---|---|---:|
| `probegen-data-nyuv2.tar.gz` | NYUv2 | 275 MB |
| `probegen-data-diode.tar.gz` | DIODE | 1.7 GB |
| `probegen-data-kitti.tar.gz` | KITTI | 466 MB |
| `probegen-data-refcoco.tar.gz` | RefCOCO, RefCOCO+, RefCOCOg | 157 MB |
| `probegen-data-ReasonSeg.tar.gz` | ReasonSeg | 913 MB |
| `probegen-data-DRAM.tar.gz` | DRAM | 20 MB |
| `probegen-data-pixmo-points.tar.gz` | PixMo-Points | 134 MB |

```bash
hf download Becomebright/ProbeGen --repo-type dataset --local-dir . \
  --include "*.tar.gz" --include SHA256SUMS      # or download from the web page
mkdir -p data
for archive in probegen-data-*.tar.gz; do tar -xzf "$archive" -C data; done
```

ScanNet and Manga109 are not included because their terms forbid
redistribution; obtain them as described below.

### ScanNet

Request access and agree to the [ScanNet Terms of Use](http://www.scan-net.org/).
Then download the preprocessed validation split prepared by
[Marigold](https://github.com/prs-eth/Marigold) and extract it:

```bash
wget https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/scannet/scannet_val_sampled_800_1.tar
mkdir -p data/depth/scannet
tar -xf scannet_val_sampled_800_1.tar -C data/depth/scannet
```

### Manga109

Request access on the [Manga109 website](http://www.manga109.org/en/download.html)
and link the release's `images/` folder:

```bash
mkdir -p data/ood_seg/Manga109
ln -s /path/to/Manga109_released_<date>/images data/ood_seg/Manga109/images
```

## Layout

```text
data/
├── depth/
│   ├── nyuv2/test/<scene>/{rgb,depth}_<id>.png
│   ├── diode/{indoors,outdoor}/scene_*/scan_*/<id>{.png,_depth.npy,_depth_mask.npy}
│   ├── scannet/<scene>/{color/<id>.jpg,depth/<id>.png}
│   ├── kitti/2011_09_26/<drive>/image_02/data/<id>.png
│   └── kitti/<drive>/proj_depth/groundtruth/image_02/<id>.png
├── refseg/coco2014/train2014/COCO_train2014_<id>.jpg
├── reason_seg/ReasonSeg/test/<id>.{jpg,json}
├── ood_seg/
│   ├── DRAM/{test,labels}/<style>/<artist>/<name>.{jpg,png}
│   └── Manga109/images/<title>/<page>.jpg
└── counting/pixmo-points/images/<index>.jpg
```

## Sources and terms

| Benchmark | Source |
|---|---|
| NYUv2, DIODE, ScanNet, KITTI | Preprocessed test splits from [Marigold](https://github.com/prs-eth/Marigold); [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [DIODE](https://diode-dataset.org/), [ScanNet](http://www.scan-net.org/), [KITTI](https://www.cvlibs.net/datasets/kitti/) |
| RefCOCO, RefCOCO+, RefCOCOg | [MS-COCO 2014](https://cocodataset.org/) images, [REFER](https://github.com/lichengunc/refer) annotations |
| ReasonSeg | [LISA](https://github.com/dvlab-research/LISA) |
| DRAM | [DRAM](https://nadavc220.github.io/ssiap/) |
| Manga109 | [Manga109](http://www.manga109.org/en/), masks from [MangaSegmentation](https://huggingface.co/datasets/MS92/MangaSegmentation) |
| PixMo-Points | [allenai/pixmo-points](https://huggingface.co/datasets/allenai/pixmo-points) |

Each dataset remains subject to its original license or terms of use; please cite
the original datasets. The archives are provided for non-commercial research
evaluation. Rights holders who want material removed can contact
dishangzhe@sjtu.edu.cn.
