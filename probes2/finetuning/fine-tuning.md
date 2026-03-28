# Probe Fine-Tuning Guide
You can finetune a Modaic probe to increase its performance on a specific dataset. You can finetune a probe without any annotated data, although annotations are encouraged. Here is how modaic will fine tune the probe based on whats available in your dataset. The training will happen in 3 stages.

1. First Pass: Self-Consistency - Fine-tune on all rows with `sc_confidence`
2. Second Pass: Cross-Consistency - Fine-tune on all rows with `cc_confidence`
3. Third Pass: Hard Label - Fine Tune on all rows with `correct` or `ground_truth` label

## Instructions 
### Step 1.
Follow the [LLM Inference Guide](../inference/llm_inference.md) to generate your dataset. You can add annotations to your dataset in 4 ways
- Manual preprocessing to add the `ground_truth` column. This column should contain a json with ground truth labels for each output field in the `arbiter.yaml`'s `outputs` section
- Manual post-processing to add the `correct` column. This is a binary column indicating whether the arbiter got that row correct or not
- Unsupervised confidence estimation using self-consistency. Run `predict` with the `--self-consistency` flag. (see [LLM Inference Advanced Options Guide](../inference/llm_inference.md#advanced_options))
- Unsupervised confidence estimation using cross-consistency from a council of frontier models. Run `predict` with the `--cross-consistency` flag. (see [LLM Inference Advanced Options Guide](../inference/llm_inference.md#advanced_options))

## Step 2.
Run the dataset through the `reflect` cli with the `--with-embeddings` flag to get embeddings for each row. 
## Step 3.
Pass the dataset you got from `predict` throught the tune cli. You can specify a tune config.yaml to ovveride the default hyper parameters
`tune --model modaic/Qwen3.5-4B-probe --dataset <your output dataset> --checkpoint tuned_head.pt --config < tune config.yaml>`

If `--config` is omitted, `probes2/finetuning/default_tune.yaml` is used automatically. The root config keys define defaults for the full run, while the `self_consistency`, `cross_consistency`, and `annotated` sections override those defaults for their respective stages.

The `--checkpoint` output is a `.pt` file containing only the tuned linear head over the reflected embeddings. You can pass that checkpoint back into `reflect` to score future datasets with the tuned head.

## Example
