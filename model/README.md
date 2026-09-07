# NLP classifier training track

**Not built yet.** Digital Detective classifies artifacts with rules + keywords
+ NER today and works without anything in here.

This directory is where the optional machine-learning classifier is trained. The
full method — where the data comes from, the weak-supervision approach, the
step-by-step plan, and how the finished model plugs back in — is in
**[../docs/MODEL_TRAINING.md](../docs/MODEL_TRAINING.md)**.

```
   corpus/    M1  raw dd-collect output, parsed artifacts.jsonl, API-schema filler
   labeling/  M2  labelling functions + label-model training → weak labels
   synth/     M3  LLM prompts, Faker templates, generated rows for rare classes
   eval/      M4  stratified sample + hand labels (the gold evaluation set)
   train/     M5  training script, checkpoints, ONNX export, metrics
```

Corpus data, model checkpoints, and `*.onnx` / `*.safetensors` are git-ignored.
