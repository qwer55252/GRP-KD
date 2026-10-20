# Published results

The values below are transcribed from the ICASSP 2026 camera-ready paper. WER is reported in percent; lower is better. These are reference values, **not rerun measurements**. No released checkpoint is available; train ver5 to obtain one locally.

## Table 1: LibriSpeech in-domain and GigaSpeech out-of-domain

### Trained on LibriSpeech 100h

| Model | LS dev-clean | LS dev-other | LS test-clean | LS test-other | GS dev | GS test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher | 12.39 | 30.20 | 12.73 | 30.50 | 47.62 | 45.28 |
| Student | 16.11 | 35.46 | 16.65 | 36.31 | 51.28 | 48.91 |
| Layerwise KD | 15.13 | 38.21 | 15.57 | 39.78 | 52.90 | 50.67 |
| Logit KD | 13.53 | 31.53 | 13.71 | 32.57 | 47.54 | 45.23 |
| DiffKD | 11.69 | 28.87 | 11.94 | 29.62 | 45.79 | 43.25 |
| FM-KT | 11.48 | 28.34 | 11.68 | 29.24 | 44.39 | 42.40 |
| GRP-KD | 11.18 | 27.22 | 11.42 | 27.79 | 44.19 | 41.76 |

### Trained on LibriSpeech 960h

| Model | LS dev-clean | LS dev-other | LS test-clean | LS test-other | GS dev | GS test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher | 6.44 | 15.69 | 6.61 | 15.85 | 34.67 | 32.77 |
| Student | 10.15 | 21.32 | 10.44 | 21.69 | 40.44 | 38.20 |
| Layerwise KD | 12.16 | 24.09 | 12.26 | 24.55 | 43.07 | 40.76 |
| Logit KD | 9.34 | 19.51 | 9.56 | 19.80 | 38.12 | 35.42 |
| DiffKD | 7.31 | 16.86 | 7.49 | 17.16 | 36.08 | 34.13 |
| FM-KT | 7.81 | 17.78 | 7.99 | 17.92 | 36.62 | 34.69 |
| GRP-KD | 7.02 | 16.31 | 7.14 | 16.49 | 35.58 | 33.37 |

## Table 2: GigaSpeech XS domain shift

The student is distilled on GS XS (10h). This is a separate experiment from the LibriSpeech recipes.

| Model | GS dev | GS test | LS dev-clean | LS dev-other | LS test-clean | LS test-other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logit KD | 58.99 | 58.63 | 72.26 | 81.41 | 71.56 | 82.13 |
| Layerwise KD | 58.59 | 58.12 | 46.11 | 61.48 | 46.19 | 62.62 |
| DiffKD | 54.28 | 53.78 | 43.40 | 61.80 | 44.05 | 62.60 |
| FM-KT | 51.93 | 51.71 | 38.38 | 54.93 | 38.65 | 55.76 |
| GRP-KD | 47.48 | 47.17 | 35.16 | 52.45 | 35.99 | 52.72 |

## Table 3: Component ablation

AE: autoencoder; FM: flow matching; DF: diffusion. Columns follow dev-clean, dev-other, test-clean, test-other.

| Components | LS100 d-c | LS100 d-o | LS100 t-c | LS100 t-o | LS960 d-c | LS960 d-o | LS960 t-c | LS960 t-o |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AE | 11.44 | 28.56 | 11.77 | 29.48 | 7.13 | 16.84 | 7.26 | 16.91 |
| FM | 11.48 | 28.34 | 11.68 | 29.24 | 7.81 | 17.78 | 7.99 | 17.92 |
| DF | 11.71 | 28.62 | 11.96 | 29.39 | 7.82 | 18.00 | 7.92 | 17.82 |
| AE + FM | 11.18 | 27.87 | 11.40 | 28.91 | 7.16 | 16.56 | 7.25 | 16.56 |
| AE + DF | 11.69 | 28.87 | 11.94 | 29.62 | 7.31 | 16.86 | 7.49 | 17.16 |
| AE + FM + DF | 11.18 | 27.22 | 11.42 | 27.79 | 7.02 | 16.31 | 7.14 | 16.49 |
