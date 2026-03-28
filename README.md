# Phinglish: Phi-4 Mini Fine-Tuning for Hinglish Chat

This project fine-tunes Phi-4 Mini on Hinglish-style conversational data using Unsloth + TRL.

The workflow is notebook-first:

1. Build JSONL training/validation data in [data.ipynb](data.ipynb).
2. Fine-tune with LoRA in [finetune.ipynb](finetune.ipynb).
3. Save adapter and merged model artifacts for inference.

## Project Structure

- [data.ipynb](data.ipynb): Data preparation and JSONL export.
- [finetune.ipynb](finetune.ipynb): LoRA fine-tuning, evaluation, and model export.
- [environment.yml](environment.yml): Conda environment definition.
- [hinglish_conversations.csv](https://huggingface.co/datasets/Abhishekcr448/Hinglish-Everyday-Conversations-1M): CSV source dataset.
- [train-00000-of-00001.parquet](https://huggingface.co/datasets/manishiitg/aditi-syn-v1): Parquet source dataset.
- finetune_messages_train.jsonl: Training split.
- finetune_messages_val.jsonl: Validation split.
- outputs: Trainer checkpoints and logs.
- phi4-mini-phinglish-lora: LoRA adapter output.
- phi4-mini-phinglish-merged: Merged 16-bit model output.

## Environment Setup

Create and activate the conda environment:

```powershell
conda env create -f environment.yml
conda activate phinglish
```

Notable versions in the environment include:

- Python 3.10.19
- torch 2.10.0+cu130
- transformers 4.57.6
- datasets 4.3.0
- trl 0.24.0
- accelerate 1.12.0
- peft 0.18.1
- bitsandbytes 0.49.2
- unsloth 2026.2.1

## Data Pipeline

In [data.ipynb](data.ipynb):

1. Load datasets from CSV and Parquet.
2. Sample records:
   - 1,000 rows from CSV
   - 2,000 rows from Parquet
3. Convert CSV rows to chat-format `messages` with user/assistant roles.
4. Concatenate both sources into one dataframe.
5. Split data into train/validation (`80/20`).
6. Export:
   - `finetune_messages.jsonl`
   - `finetune_messages_train.jsonl`
   - `finetune_messages_val.jsonl`

Each JSONL row is expected to contain a `messages` field with chat turns.

## Fine-Tuning Pipeline

In [finetune.ipynb](finetune.ipynb):

1. Load base model with Unsloth:
   - `unsloth/Phi-4-mini-instruct`
   - `max_seq_length=2048`
   - `load_in_4bit=True`
2. Configure LoRA:
   - `r=16`, `lora_alpha=16`, `lora_dropout=0`
   - target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
3. Load JSONL train/val splits via `datasets.load_dataset`.
4. Convert `messages` to text via tokenizer chat template.
5. Train with `trl.SFTTrainer`.

Current training arguments in notebook:

- `output_dir="./outputs"`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`
- `num_train_epochs=2`
- `learning_rate=2e-4`
- `eval_steps=50`, `save_steps=50`
- `save_total_limit=2`
- `optim="paged_adamw_8bit"`

## Produced Artifacts

After training, notebook cells save:

- LoRA adapter + tokenizer in `phi4-mini-phinglish-lora`
- Merged 16-bit model + tokenizer in `phi4-mini-phinglish-merged`
- Trainer outputs/checkpoints in `outputs`

The notebook also includes a quick generation example for Hinglish responses.

## Typical Run Order

1. Activate environment.
2. Run all cells in [data.ipynb](data.ipynb).
3. Verify train/val JSONL files are generated.
4. Run all cells in [finetune.ipynb](finetune.ipynb).
5. Check [outputs](outputs), [phi4-mini-phinglish-lora](phi4-mini-phinglish-lora), and [phi4-mini-phinglish-merged](phi4-mini-phinglish-merged).

## Notes

- Training is configured for CUDA and 4-bit loading.
- [finetune.ipynb](finetune.ipynb) disables some compile/dynamo paths for stability with Phi longrope.
- Keep enough GPU memory for your sequence length and batch configuration.

## Troubleshooting

- If conda fails to solve environment, update conda and retry:

```powershell
conda update -n base -c defaults conda
conda env create -f environment.yml
```

## References & Citations

### Dataset: Hinglish Everyday Conversations

This project utilizes the 21M Hinglish dataset for training/evaluation.

```bibtex
@misc{Hinglish-Chat-21M,
  author = {Abhishek Khatri},
  title = {Hinglish Everyday Conversations Dataset},
  year = {2024},
  url = {[https://github.com/Abhishekcr448/Hinglish-Chat-21M](https://github.com/Abhishekcr448/Hinglish-Chat-21M)},
}
```

### Library: Transformer Reinforcement Learning (TRL)

The training pipeline was built using the TRL library from Hugging Face.

```bibtex
@misc{vonwerra2022trl,
  title        = {{TRL: Transformer Reinforcement Learning}},
  author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
  year         = 2020,
  journal      = {GitHub repository},
  publisher    = {GitHub},
  howpublished = {\url{[https://github.com/huggingface/trl](https://github.com/huggingface/trl)}}
}
```
