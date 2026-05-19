import asyncio
import yaml
import json
import torch
import random
from typing import List
from vllm import LLM, SamplingParams

# vLLM LoRA
try:
    from vllm.lora.request import LoRARequest
except Exception:
    LoRARequest = None


# ============================================================
# Sampling
# ============================================================

def sample(
    llm,
    conversations,
    temperature=1.0,
    top_p=1.0,
    max_tokens=6000,
    stop=[],
    min_tokens=1,
    lora_request=None,
):
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
    )

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for messages in conversations
    ]

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )

    return [o.outputs[0].text for o in outputs]


# ============================================================
# Question abstraction
# ============================================================

class Question:
    def __init__(
        self,
        id: str,
        paraphrases: List[str],
        temperature: float = 1.0,
        system: str = None,
        **ignored,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system

    def get_input(self, n_per_question):
        chosen = random.choices(self.paraphrases, k=n_per_question)
        conversations = []

        for text in chosen:
            messages = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            messages.append({"role": "user", "content": text})
            conversations.append(messages)

        return chosen, conversations


# ============================================================
# Load model
# ============================================================

def load_model(model_path):
    return LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
        enable_lora=True,       
        max_lora_rank=128,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        max_num_seqs=32,
    )


# ============================================================
# Load questions
# ============================================================

def load_questions(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    questions = []
    for q in data:
        assert q["type"] == "free_form_judge_0_100"
        questions.append(Question(**q))
    return questions


# ============================================================
# Main generation logic
# ============================================================

async def generate_responses(
    model_path,
    lora_path,
    questions_path,
    n_per_question,
    output_path,
):
    print(f"[Load] base model: {model_path}")
    llm = load_model(model_path)

    print(f"[Load] questions: {questions_path}")
    questions = load_questions(questions_path)

    if lora_path:
        assert LoRARequest is not None, "Your vLLM version does not support LoRA."
        lora_request = LoRARequest(
            lora_name="eval_lora",
            lora_path=lora_path,
            lora_int_id=1,
        )
        print(f"[LoRA] using checkpoint: {lora_path}")
    else:
        lora_request = None

    all_records = []

    for q in questions:
        paraphrases, conversations = q.get_input(n_per_question)

        answers = sample(
            llm,
            conversations,
            temperature=q.temperature,
            lora_request=lora_request,
        )

        for p, a in zip(paraphrases, answers):
            all_records.append({
                "question_id": q.id,
                "paraphrase": p,
                "answer": a,
            })

    print(f"[Save] {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("✅ Generation finished.")


# ============================================================
# CLI
# ============================================================

def main(
    model: str,
    questions: str,
    output: str,
    lora: str = None,
    n_per_question: int = 30,
):
    asyncio.run(
        generate_responses(
            model_path=model,
            lora_path=lora,
            questions_path=questions,
            n_per_question=n_per_question,
            output_path=output,
        )
    )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
