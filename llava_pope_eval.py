#!/usr/bin/env python3
"""
Evaluate a LLaVA model (base or DPO-trained) on the POPE benchmark.

This script:
- Loads either the base LLaVA model or a DPO-adapted model
- Loads POPE benchmark questions
- Runs inference with images + questions
- Saves predictions to JSONL
- Optionally evaluates simple accuracy metrics
"""

import argparse
import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel


def load_model(model_name, dpo_checkpoint=None, device="cuda"):
    """
    Load either base model or DPO adapter on top of base model.
    """
    print(f"Loading base model: {model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if dpo_checkpoint:
        print("==============================================================")
        print(f"Loading DPO adapter from: {dpo_checkpoint}")
        print("==============================================================")
        model = PeftModel.from_pretrained(model, dpo_checkpoint)

    model.eval()
    return model.to(device)


def evaluate_pope(model, processor, pope_path, coco_path, set_name, type_name, output_dir="./results"):
    """
    Run inference on the POPE benchmark dataset.
    """
    # Determine dataset file
    if set_name == "random":
        questions_file = os.path.join(pope_path, "output/coco/coco_pope_random.json")
    elif set_name == "popular":
        questions_file = os.path.join(pope_path, "output/coco/coco_pope_popular.json")
    elif set_name == "adv":
        questions_file = os.path.join(pope_path, "output/coco/coco_pope_adversarial.json")
    else:
        raise ValueError(f"Unknown set: {set_name}")

    # Load questions
    with open(questions_file, "r") as f:
        lines = f.readlines()

    results = []
    print("==============================================================")
    for line in tqdm(lines, desc=f"Generating on {set_name}"):
        data = json.loads(line)
        question = data["text"]
        image_name = data["image"]
        question_id = data["question_id"]
        label = data["label"]

        # Load image
        image_path = os.path.join(coco_path, "val2014", image_name)
        image = Image.open(image_path).convert("RGB")

        # Prepare prompt
        chat_data = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        prompt = processor.apply_chat_template(chat_data, add_generation_prompt=True)

        # Process inputs
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        ).to("cuda")

        # Generate answer
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        results.append({
            "question_id": question_id,
            "question": question,
            "answer": output_text,
            "label": label
        })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"pope_{set_name}_{type_name}.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved predictions to {out_file}")
    return results

def main(args):

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name, do_image_splitting=False)

    model = load_model(args.model_name, args.dpo_checkpoint)

    results = evaluate_pope(
        model=model,
        processor=processor,
        pope_path=args.pope_path,
        coco_path=args.coco_path,
        set_name=args.set_name,
        type_name=args.model_type,
        output_dir=args.output_dir
    )
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA model on POPE benchmark")
    parser.add_argument("--model_type", type=str, help="Which model to evaluate", choices=["base", "dpo"], default="base")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--dpo_checkpoint", type=str, default=None, help="Path to DPO adapter checkpoint")
    parser.add_argument("--pope_path", type=str, required=True, help="Path to POPE dataset folder")
    parser.add_argument("--coco_path", type=str, required=True, help="Path to COCO images")
    parser.add_argument("--set_name", type=str, default="random", choices=["random", "popular", "adv"],
                        help="Which POPE split to evaluate")
    
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    args = parser.parse_args()
    main(args)

    