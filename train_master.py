import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import evaluate


# 1.cpu optimising 
#  definning   CPU cores can 
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


# 2. Dividing problem into small PROPER FUNCTIONS 

def transform(example_batch):
    inputs = processor([x.convert("RGB") for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    prec = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    rec = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    
    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1["f1"]
    }

# fixing func and removing lambda func
def custom_collate_fn(batch):
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch])
    }


#3.using loop for training dataset
if __name__ == '__main__':
    print("1. Loading the Full Food-101 Dataset...")
    dataset = load_dataset("food101")

    labels = dataset["train"].features["label"].names
    id2label = {str(i): c for i, c in enumerate(labels)}
    label2id = {c: str(i) for i, c in enumerate(labels)}

    print("2. Initializing Vision Transformer (ViT)...")
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    print("3. Pre-Processing Images for GPU...")
    prepared_ds = dataset.with_transform(transform)

    print("4. Configuring MAX POWER RTX 4050 Parameters...")
    training_args = TrainingArguments(
        output_dir="./food_master_model",
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,    
        fp16=True,                        
        dataloader_num_workers=4,         
        
        num_train_epochs=3,               
        learning_rate=5e-5,               
        warmup_steps=500,                 
        lr_scheduler_type="cosine",       
        
        eval_strategy="epoch",            
        save_strategy="epoch",            
        save_total_limit=2,               
        load_best_model_at_end=True,      
        metric_for_best_model="f1",       
        
        logging_steps=50,
        remove_unused_columns=False,
        report_to="none"                  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=custom_collate_fn,  
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] 
    )

    print("5. 🚀 IGNITING 3-EPOCH MASTER TRAINING...")
    trainer.train()

    print("6. Training Complete! Saving the absolute best model...")
    trainer.save_model("./food_master_model/best_model")
    processor.save_pretrained("./food_master_model/best_model")

    print("✅ Master Model safely saved to './food_master_model/best_model'!")