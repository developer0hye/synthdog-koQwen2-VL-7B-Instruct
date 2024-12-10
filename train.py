import os
os.environ["HF_HOME"] = "/workspace/"

from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info

from transformers import TextStreamer
from datasets import load_dataset
import json

from functools import partial


# Load datasets
train_dataset = load_dataset("naver-clova-ix/synthdog-ko", split="train")
NUM_TRAIN_SAMPLES = 50000
train_dataset = train_dataset.shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))

validation_dataset = load_dataset("naver-clova-ix/synthdog-ko", split="validation")

# Constants
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 960 * 28 * 28
INSTRUCTION = """
이미지에 포함된 문자를 추출하세요. 출력은 JSON 형식으로 제공하며, 키는 "text_sequence"이고 값은 추출된 문자입니다.\n
예시: {"text_sequence": 인식한 문자}
"""

def convert_to_conversation(sample, instruction, min_pixels, max_pixels):
    ground_truth_dict = json.loads(sample["ground_truth"])
    ground_truth = ground_truth_dict["gt_parse"]["text_sequence"]
    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": instruction},
             {"type": "image", "image": sample["image"],
              "min_pixels": min_pixels,
              "max_pixels": max_pixels,
              }]
         },
        {"role": "assistant",
         "content": [
             {"type": "text", "text": '{"text_sequence": "' + ground_truth + '"}'}]
         },
    ]
    return {"messages": conversation}

formatting_func = partial(convert_to_conversation, instruction=INSTRUCTION, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

# Model setup
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=64,
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

FastVisionModel.for_inference(model)

def run_inference(dataset, formatting_func, num_samples, filename, filemode, mode):
    with open(filename, filemode, encoding="utf-8") as f:
        f.write(f"{mode} training\n")
        for i in range(min(num_samples, len(dataset))):
            data = dataset[i]
            formatted_data = formatting_func(data)
            messages = formatted_data["messages"]
            prompt_without_ground_truth = [messages[0]]
            ground_truth = messages[1]['content'][0]['text']

            text = tokenizer.apply_chat_template(prompt_without_ground_truth, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(prompt_without_ground_truth)

            inputs = tokenizer(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")

            text_streamer = TextStreamer(tokenizer, skip_prompt=True)
            generated_ids = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1024, use_cache=True)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            f.write(f'label: {ground_truth}\noutput: {output_text}\n')
            f.write('-----------------------------------\n')
            f.flush()

run_inference(validation_dataset, formatting_func, num_samples=10, filename="results_train.txt", filemode="w", mode="before")

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer, formatting_func),
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=1024,
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("synthdog-koqwen2-vl-7b-instruct-lora-model")
tokenizer.save_pretrained("synthdog-koqwen2-vl-7b-instruct-lora-model")

FastVisionModel.for_inference(model)

run_inference(validation_dataset, formatting_func, num_samples=10, filename="results_train.txt", filemode="a", mode="after")