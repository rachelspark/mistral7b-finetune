from modal import Image, Stub, Volume, Secret

BASE_MODEL = "mistralai/Mistral-7B-v0.1"


# def download_models():
#     import torch
#     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config)
#     tokenizer = AutoTokenizer.from_pretrained(
#         BASE_MODEL,
#         model_max_length=512,
#         padding_side="left",
#         add_eos_token=True
#     )
#     tokenizer.pad_token = tokenizer.eos_token


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
        "bitsandbytes",
        "git+https://github.com/huggingface/peft.git",
        "git+https://github.com/huggingface/transformers.git",
        "git+https://github.com/huggingface/accelerate.git@11e2e99cfca3afe1cefe02111f40665b692b86fb",
        "datasets",
        "scipy",
        "ipywidgets",
        "wandb",
    )
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    # .pip_install("huggingface_hub==0.17.1", "hf-transfer==0.1.3")
    # .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)


stub = Stub(name="example-mistral-7b-finetune", image=image, secrets=[Secret.from_name("huggingface")])

stub.pretrained_volume = Volume.persisted("example-pretrained-vol")
stub.results_volume = Volume.persisted("example-results-vol")

model_store_path = "/vol/models"

VOLUME_CONFIG = {
    "/pretrained": stub.pretrained_volume,
    "/results": stub.results_volume,
}