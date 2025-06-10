from model import SmolLM_135M
import os


def hello_world_demo(device):
    model, tokenizer = SmolLM_135M(device)

    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    outputs = model.generate(inputs)

    print(tokenizer.decode(outputs[0]))


def eval_mmlu_pro():
    # Run script
    script_path = os.path.join("benchmarks", "MMLU-Pro", "scripts", "SmolLM", "eval_vanilla.sh")
    os.system(f"sh {script_path}")


def main():
    device = "cuda:1"

    # hello_world_demo(device)
    eval_mmlu_pro()


if __name__ == "__main__":
    main()
