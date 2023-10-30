from pathlib import Path
from modal import Image, Stub, Volume, NetworkFileSystem, Secret

BASE_MODEL = "mistralai/Mistral-7B-v0.1"

MODEL_PATH = Path("/model")

def download_models():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.save_pretrained(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=512,
        padding_side="left",
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(MODEL_PATH)


image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.8",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "bitsandbytes==0.41.1",
        "git+https://github.com/huggingface/peft.git@2464c572eba6b60a9d19ba1913fcec6bc0a2724b",
        "git+https://github.com/huggingface/transformers.git@a2f55a65cd0eb3bde0db4d5102a824ec96c7d7e9",
        "git+https://github.com/huggingface/accelerate.git@11e2e99cfca3afe1cefe02111f40665b692b86fb",
        "datasets==2.14.6",
        "scipy==1.11.3",
        "wandb==0.15.12",
    )
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    .run_function(download_models)
)

stub = Stub(name="example-mistral-7b-finetune", image=image)

stub.training_data_volume = Volume.persisted("training-data-vol")
# stub.pretrained_volume = Volume.persisted("pretrained-vol")
stub.results_volume = Volume.persisted("results-vol")

VOLUME_CONFIG = {
    "/training_data": stub.training_data_volume,
    "/results": stub.results_volume,
}