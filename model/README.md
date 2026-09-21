# NLP classifier training track (category model)

**Not built yet.** This directory is where the optional CATEGORY-predicting
model would be trained. Digital Detective classifies artifacts with rules +
keywords + NER for category today, no model needed. The full method — where
the data comes from, the weak-supervision approach, the step-by-step plan,
and how the finished model plugs back in — is in
**[../docs/MODEL_TRAINING.md](../docs/MODEL_TRAINING.md)**.

A **different** model — binary relevant/not_relevant, not a category — IS
trained and in use, but by a standalone script outside this repo entirely,
not this directory. See
[../docs/MODEL_TRAINING.md#the-relevance-model-trained-and-in-use](../docs/MODEL_TRAINING.md#the-relevance-model-trained-and-in-use).

```
   corpus/    M1  raw dd-collect output, parsed artifacts.jsonl, API-schema filler
   labeling/  M2  labelling functions + label-model training → weak labels
   synth/     M3  LLM prompts, Faker templates, generated rows for rare classes
   eval/      M4  stratified sample + hand labels (the gold evaluation set)
   train/     M5  training script, checkpoints, ONNX export, metrics
```

Corpus data, model checkpoints, and `*.onnx` / `*.safetensors` are git-ignored.
