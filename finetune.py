from modal import Secret
from transformers import TrainerCallback

from common import stub, BASE_MODEL, VOLUME_CONFIG

WANDB_PROJECT = "huggingface-mistral7b"

class CheckpointCallback(TrainerCallback):
    def __init__(self, volume):
        self.volume = volume

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print("running commit on modal.Volume after model checkpoint")
            self.volume.commit()

# @stub.function(
#     volumes=VOLUME_CONFIG,
#     memory=1024 * 100,
#     timeout=3600 * 4,
# )
# def download_model(model_name: str):
#     from huggingface_hub import snapshot_download
#     from transformers.utils import move_cache

#     try:
#         file_path = snapshot_download(model_name, local_files_only=True)
#         print(f"Volume contains {model_name} at {file_path}.")
#     except FileNotFoundError:
#         print(f"Downloading {model_name} (no progress bar) ...")
#         snapshot_download(model_name)
#         move_cache()

#         print("Committing /pretrained directory (no progress bar) ...")
#         stub.pretrained_volume.commit()

@stub.function(
    gpu="A100",
    secret=Secret.from_name("my-wandb-secret"),
    timeout=60 * 60 * 4,
    volumes=VOLUME_CONFIG,
    cloud="oci"
)
def finetune(model_name: str, resume_from_checkpoint: str = None):
    import os
    from datetime import datetime

    import torch
    import transformers
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset

    train_dataset = load_dataset('gem/viggo', split='train')
    eval_dataset = load_dataset('gem/viggo', split='validation')
    test_dataset = load_dataset('gem/viggo', split='test')


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
    # except FileNotFoundError:
    #     stub.pretrained_volume.reload()
    #     model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=f"./pretrained", quantization_config=bnb_config, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=512,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(prompt):
        result = tokenizer.__call__(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt =f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
        This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

        ### Target sentence:
        {data_point["target"]}

        ### Meaning representation:
        {data_point["meaning_representation"]}
        """

        return tokenize(full_prompt)
    
    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    
    # model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )


    model = get_peft_model(model, config)
    model.to("cuda")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    os.environ["WANDB_PROJECT"] = "huggingface-mistral7b"
    os.environ["WANDB_WATCH"] = "gradients"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        callbacks=[CheckpointCallback(stub.results_volume)],
        args=transformers.TrainingArguments(
            output_dir=f"/results",
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=5,
            learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
            logging_steps=1,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir=f"/results/logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=1,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=1,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to="wandb",           # Comment this out if you don't want to use weights & baises
            run_name=f"mistral7b-finetune-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"      
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )

    for batch in trainer.get_train_dataloader():
        break

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = trainer.model.to(device)(**batch)

    print("calling backward() on loss")
    loss = outputs.loss
    loss.backward()

    print("performing optimization step")
    trainer.create_optimizer()
    trainer.optimizer.step()
    print("finished step")

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    
    model.save_pretrained(f"/results")

    stub.results_volume.commit()


@stub.local_entrypoint()
def main():
    # print(f"Syncing base model {BASE_MODEL} to volume.")
    # download.remote(BASE_MODEL)

    print("Starting finetuning.")
    finetune.remote(BASE_MODEL)
    print("Done!")


