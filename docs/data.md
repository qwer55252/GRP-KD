# Data preparation

All supported entry points consume NeMo JSONL manifests. Each line describes one mono, 16 kHz utterance:

```json
{"audio_filepath": "/absolute/path/utterance.flac", "duration": 4.25, "text": "a lowercase reference transcript"}
```

Use absolute paths, positive durations in seconds, and lowercase transcripts without surrounding whitespace. Training and evaluation validate file existence and manifest structure before importing the model stack. The LibriSpeech builder additionally checks sample rate and channel count. A manifest does not contain the audio itself.

## LibriSpeech 100h

Download `train-clean-100`, `dev-clean`, `dev-other`, `test-clean`, and `test-other` from [OpenSLR](https://www.openslr.org/12). Extract them into one `LibriSpeech` directory. Run:

```bash
python cmd/prepare_librispeech.py --root /path/to/LibriSpeech \
  --splits train-clean-100 dev-clean dev-other test-clean test-other \
  --output-dir data/manifests
```

Training uses `train-clean-100`; validation uses `dev-clean`. The other splits are evaluated separately and must never be concatenated into training data.

## LibriSpeech 960h

Also download `train-clean-360` and `train-other-500`:

```bash
python cmd/prepare_librispeech.py --root /path/to/LibriSpeech \
  --splits train-clean-100 train-clean-360 train-other-500 \
  --output-dir data/manifests
cat data/manifests/train-clean-100.jsonl \
    data/manifests/train-clean-360.jsonl \
    data/manifests/train-other-500.jsonl > data/manifests/train-960.jsonl
```

The saved LS960 recipe differs from LS100 through this combined training manifest. Model settings and validation split are the same.

## GigaSpeech

Obtain data through the [official GigaSpeech project](https://github.com/SpeechColab/GigaSpeech), following its access and usage terms. For Table 1, prepare separate dev/test manifests. For Table 2, the paper uses the XS 10-hour training subset.

The original GigaSpeech evaluator removes punctuation tags `<COMMA>`, `<PERIOD>`, `<QUESTION_MARK>`, and `<EXCLAMATION_MARK>`, drops tag-only references, lowercases text, and excludes utterances shorter than one second. Its full preparation logic is retained in the [historical evaluator](https://github.com/qwer55252/GRP-KD/blob/b995ba3/asr_inference_diffm.py). Exact segment selection and normalization need to match that script before comparing WER; generic transcripts are not sufficient to claim paper reproduction. This release does not yet supply a validated replacement GigaSpeech converter.

Once validated manifests are available:

```bash
python -m grp_kd.evaluate --model outputs/ver5-ls100/student.nemo \
  --manifest gs-dev=/path/to/gs-dev.jsonl \
  --manifest gs-test=/path/to/gs-test.jsonl \
  --output outputs/ver5-ls100/gigaspeech.json
python cmd/compare_results.py \
  --results outputs/ver5-ls100/gigaspeech.json --training-set ls100
```
