from transformers import AutoModelForCausalLM, AutoTokenizer


def SmolLM_135M(device):
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    return model, tokenizer
