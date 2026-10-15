# ProbeGen

**Are Image Generators Zero-Shot Perceivers? A Rigorous Evaluation** (BMVC 2026)
Shangzhe Di, Zhaokai Wang, Weidi Xie — Shanghai Jiao Tong University

ProbeGen poses depth estimation, referring/reasoning segmentation and object
counting as text-prompted image generation, and scores the generated images on
11 public benchmarks. This repository provides the benchmark data and the code
to generate and evaluate predictions with the paper's default settings.

| Task | Benchmarks | Metrics |
|---|---|---|
| Monocular depth | NYUv2, DIODE, ScanNet, KITTI | AbsRel |
| Referring segmentation | RefCOCO, RefCOCO+, RefCOCOg | gIoU, cIoU |
| Reasoning segmentation | ReasonSeg | gIoU, cIoU |
| OOD segmentation | DRAM, Manga109 | gIoU, cIoU |
| Counting | PixMo-Points | MAE, RMSE |

## Setup

```bash
git clone https://github.com/Becomebright/ProbeGen.git && cd ProbeGen
pip install -e .
```

Download the data into `data/` as described in [DATA.md](DATA.md).

## Generate

Copy the API template and add your key (`api.local.yaml` is git-ignored):

```bash
cp api.sample.yaml api.local.yaml
python -m probegen_eval generate --dataset ReasonSeg --model nano-banana-2
```

| `--model` | Model ID | API (`api.local.yaml` section) |
|---|---|---|
| `nano-banana-2` | `gemini-3.1-flash-image-preview` | Gemini (`gemini`) |
| `nano-banana-pro` | `gemini-3-pro-image-preview` | Gemini (`gemini`) |
| `nano-banana-2.1` | `gemini-nano-banana-2.1` | Gemini (`gemini`) |
| `gpt-image-2` | `gpt-image-2` | OpenAI image edits (`openai`) |
| `seedream-5.0` | `doubao-seedream-5-0-260128` | Volcengine Ark image generation (`seedream`) |
| `gemini-3.5-flash` | `gemini-3.5-flash` | Gemini (`gemini`), counting only |
| `gpt-5.4` | `gpt-5.4` | OpenAI chat completions (`openai`), counting only |
| `claude-opus-4.7` | `claude-opus-4-7` | Anthropic messages (`anthropic`), counting only |

The last three are MLLMs: on PixMo-Points they answer with a number (saved as
`.txt`) instead of an image. Predictions are saved to `outputs/<model>/<dataset>/`. Rerunning the command
skips existing predictions and retries failed ones. Use `--n-samples 5` for a
quick test.

Default settings from the paper: the input is resized to a 1024-pixel long side
and sent as JPEG together with the task prompt (mid version, below).

| Task | Prompt |
|---|---|
| Referring segmentation | `Segment {label}. Make the target object pure white and everything else pure black.` |
| Reasoning segmentation | `{question}` + `Create a binary segmentation mask for the target object that satisfies the above description. Make the target object pure white and everything else pure black.` |
| OOD segmentation | `Segment all {label}. Make the target objects (one or more) pure white and everything else pure black.` |
| Depth | `Create a grayscale depth map of this image: brighter = closer, darker = farther.` |
| Counting | `Put one small solid red dot precisely on each {label} in this image. Use the same bright red color for every dot. Do not change anything else in the image.` |
| Counting (MLLMs) | `Count all objects of {label} in the image. Answer with a single integer only. Do not include words, punctuation, reasoning steps, or explanations. If uncertain, guess one integer.` |

## Evaluate

```bash
python -m probegen_eval evaluate --dataset ReasonSeg \
  --predictions outputs/nano-banana-2/ReasonSeg --output reports/reasonseg.json
```

Predictions from other models can be evaluated in the same way: save one image
per sample in a folder, named by the `prediction` field of
`manifests/<dataset>.json`. Missing predictions are excluded from the metrics
and listed in the report.

- **Segmentation:** the prediction is converted to grayscale, resized to the
  ground-truth resolution and binarized with Otsu's threshold. gIoU is the mean
  per-image IoU, cIoU the cumulative intersection over cumulative union.
- **Depth:** the grayscale prediction (brighter = closer) is aligned to the
  ground truth by least-squares scale and shift in disparity space.
- **Counting:** red dots added relative to the input are detected by color,
  difference and shape filtering, and counted. For MLLM answers (`.txt`), the
  first integer in the text is the count (0 if there is none).

Metrics are reported as fractions; multiply by 100 for the paper's tables.

## TODO

- [ ] GPT-Image-2.5 (`gpt-image-2.5-sunburst`)
- [ ] Open-weight generators run locally: FLUX.2-dev (NF4), Qwen-Image-Edit-2511
- [ ] Specialist models: Depth Anything V2, Marigold, Lotus-2, SAM3 + Qwen3-VL-Thinking,
      Qwen3-VL-SAMTok, CountGD++

## Citation

```bibtex
@inproceedings{bmvc26_probegen,
  title={Are Image Generators Zero-Shot Perceivers? A Rigorous Evaluation},
  author={Di, Shangzhe and Wang, Zhaokai and Xie, Weidi},
  booktitle={BMVC},
  year={2026}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE). Manifests and data
keep the terms of their original datasets; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Contact: dishangzhe@sjtu.edu.cn.
