# Modaic Probes 
<p align="center">
  <img src="./assets/modaic_probes_white.png" alt="Modaic Probes" width="50%" />
</p>

This repository contains a guide for working with Modaic's Open Source confidence estimation probes. Currently, the only supported model is `Qwen/Qwen3.5-4B`.

## Inference 
There's two types of inference you will need to run to make use of the full judging lifecycle.
1. LLM Judge Inference - This is inference you will run to get the LLM's predictions for a specific classification task. Follow the [LLM Inference Guide](./probes2/inference/llm_inference.md) to get started.
2. Probe Inference - This is inference you will run on the probe to get confidence estimations for the LLM's predictions. Follow the [Probe Inference Guide](./probes2/inference/probe_inference.md) to get started.


<!-- ## Fine-tuning
You can fine-tune a probe to improve it's confidence estimation abilities for a specific classification task. Follow the [Fine-tuning Guide](./probes2/fine-tuning/fine-tuning.md) to get started. -->
