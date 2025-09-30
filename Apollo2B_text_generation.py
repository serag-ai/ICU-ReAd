#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import os
import time
import datetime
import gc
import random
import re
import transformers
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import evaluate 
from datasets import (
    Dataset,
    DatasetDict)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model, 
    PeftModel
)
from torch.utils.data import (
    Dataset)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'


w_token ='YOUR_TOKEN'
r_token ='YOUR_TOKEN'





from huggingface_hub import login
# HUGGINGFACE_TOKEN = os.environ.get("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")
# login(token=HUGGINGFACE_TOKEN)

#Loading the files (Full dataset)

def load_file(data_file):
    df = pd.read_csv(data_file)
    print('File loaded successfully')
    return df

# Load the CSV file
file_path_2 = '/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/llm.csv'
df2 = load_file(file_path_2)

#Data Pre_processing
def filter_unwanted_words(diseases, unwanted_words):
    return [disease for disease in diseases if not any(word in disease for word in unwanted_words)]

# Function to create prompts for each row
def create_prompts(df):
    prompts = []
    admission_responses = []
    
    unwanted_words = ['performed'] 
    
    disease_columns = [col for col in df.columns if col.startswith('Diagnose_')]
    
    for index, row in df.iterrows():
        icu_diseases = [disease.replace('Diagnose_', '').replace('_', ' ').lower() 
                        for disease in disease_columns if row[disease] == 1]
        icu_diseases = filter_unwanted_words(icu_diseases, unwanted_words)

        past_disease_columns = [col for col in df.columns if col.startswith('Past_') and col != 'Past_No Health Problems']
        
        if 'Past_No Health Problems' in df.columns and row['Past_No Health Problems'] == 1:
            past_diseases = "There was no prior health issue"
        else:
            past_diseases = [disease.replace('Past_', '').replace('_', ' ').replace(' 1', '').lower() 
                             for disease in past_disease_columns if row[disease] == 1]
            past_diseases = filter_unwanted_words(past_diseases, unwanted_words)
        
        num_visits = row['unitvisitnumber']
        if num_visits == 1:
            admission_type = 're-admitted'
            admission_response = 'Yes, this patient has a possibility to get re-admitted to the ICU.'
        else:
            admission_type = 'admitted once'
            admission_response = 'No, this patient is not likely to get re-admitted to the ICU.'

        # Create the patient details prompt
        comment = f"""
        Patient details:
        Gender: {row['gender']}
        Age: {row['age']}
        Ethnicity: {row['ethnicity']}
        Height: {row['admissionheight']}
        Weight: {row['admissionweight']}
        Hospital Stay Duration: {row['hospitaldischargeoffset']} minutes
        Admission Diagnosis: {row['apacheadmissiondx']}
        Admitted to the ICU from {row['unitadmitsource']}
        ICU type: {row['unittype']}
        ICU diagnoses: {', '.join(icu_diseases) if icu_diseases else 'None'}.
        Past diagnoses: {past_diseases if past_diseases else 'None'}.
        ICU Stay Duration: {row['unitdischargeoffset']} minutes

        """

        prompts.append(comment.strip())
        admission_responses.append(admission_response.strip())

    return prompts, admission_responses


#Creating the prompts and answers 
prompts, answers = create_prompts(df2)




#print(prompts[300])
#print(answers[300])

#print(prompts[14])
#print( answers[14])

#######################################################################


# commit message and model name
commit_message = "Fine-tuning FreedomIntelligence/Apollo-2B on ICU dataset with shorter prompt"

model_name = "FreedomIntelligence/Apollo-2B"


#Configurations 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#model and tokenizer loading 
model = AutoModelForCausalLM.from_pretrained(model_name, #model_name
                                             quantization_config=bnb_config,
                                             device_map="auto",# automatically figures out how to best use CPU + GPU for loading model
                                             token=r_token,
                                             trust_remote_code=False) # prevents running custom model files on your machine
                                             #revision="main") # which version of model to use in repo

tokenizer = AutoTokenizer.from_pretrained(model_name, #model_name
                                        token=r_token, add_eos_token=True, use_fast=True)


tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token


model.eval() # model in evaluation mode (dropout modules are deactivated)


#Instruction strings for the prompt
instructions_string = """
You are a virtual assistant in the intensive care unit (ICU), specialized in analyzing patient data and answering questions.

    """
output_string = """
   **Response:**
    - Analyse the patient information
    - Answer with "Yes" or "No" for likelihood of re-admission.
    - Give me label One admission or Multiple Admission
    - Explain your answer, detailing why re-admission is likely or unlikely based on the provided information.
"""
# craft prompt
input = prompts[114]

prompt =f'''<|system|>:\n {instructions_string} <|user|>: {input} \n Would this patient get re-admitted to Intensive Care Unit?\n<|assistant|>:\n '''
# generate output
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=512, temperature= 0.4, do_sample = True)
#print("without instruction")
print('------------------------------------------------')
print(tokenizer.batch_decode(outputs)[0])


#Testing the model performance with pipeline
input = prompts[90]
pl = pipeline("text-generation", model="FreedomIntelligence/Apollo-2B", tokenizer="FreedomIntelligence/Apollo-2B", max_new_tokens=512, device_map="cuda")
question = "Would this patient get re-admitted to Intensive Care Unit?"
answer = pl(f"Context: {instructions_string}\n {input}\n\nQuestion: {question}\n\nAnswer: ")
print(answer)




#Spliting the dataset
train_summaries, temp_summaries, train_labels, temp_labels = train_test_split(
    prompts, answers, test_size=0.1, shuffle=True, random_state=0, stratify=answers
)
dev_summaries, test_summaries, dev_labels, test_labels = train_test_split(
    temp_summaries, temp_labels, test_size=0.5, shuffle=True, random_state=0, stratify=temp_labels
)


# Convert the train and dev lists to DataFrames
train = pd.DataFrame({'summary': train_summaries, 'label': train_labels})
dev = pd.DataFrame({'summary': dev_summaries, 'label': dev_labels})

# Create a DataFrame for the test set with separate columns for summaries and labels
test = pd.DataFrame({'summary': test_summaries, 'label': test_labels})

#Saving the dataset splits for productivity
train.to_csv("train_set_r32_apollo.csv")
dev.to_csv("dev_set_r32_apollo.csv")
test.to_csv("test_set_r32_apollo.csv")


# Display the sizes of each set
print(f"Train set size: {len(train)}")
print(f"Dev set size: {len(dev)}")
print(f"Test set size: {len(test)}")

#Create DatasetDict for train, dev and test
train_list = train.to_dict(orient='records')
dev_list = dev.to_dict(orient='records')
test_list = test.to_dict(orient='records')

from datasets import Dataset, DatasetDict
dataset = DatasetDict({
    'train': Dataset.from_list(train_list),
    'dev': Dataset.from_list(dev_list),
    'test': Dataset.from_list(test_list)
})



#Fine tuning the model! 

model.train() # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)

#Parameters
LORA_R = 32 
LORA_ALPHA = 16 
LORA_DROPOUT = 0.05
Target_modules = ["q_proj","v_proj","k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

# LoRA config
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules= Target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA trainable version of model
model = get_peft_model(model, config)

# trainable parameter count
# print_trainable_parameters(model)
model.print_trainable_parameters()

from datasets import Dataset, DatasetDict
train_list = train.to_dict(orient='records')
dev_list = dev.to_dict(orient='records')
dataset = DatasetDict({
    'train': Dataset.from_list(train_list),
    'dev': Dataset.from_list(dev_list)
})
dataset



#data = load_dataset('csv', data_files={'train': '/content/drive/MyDrive/Colab Notebooks/summaries_train.txt'})
example_template = lambda text, label: {
    'example': f'''<|system|>:\n {instructions_string} <|user|>: {text} \n Would this patient get re-admitted to Intensive Care Unit?\n {output_string} \n <|assistant|>:\n '''  + "\n" + label 
}

# Create a list of dictionaries for the train, dev, and test sets using the template
train_list = [example_template(train_summaries[i], train_labels[i]) for i in range(len(train_summaries))]
dev_list = [example_template(dev_summaries[i], dev_labels[i]) for i in range(len(dev_summaries))]

# Convert the lists to Hugging Face Datasets
train_dataset = Dataset.from_list(train_list)
dev_dataset = Dataset.from_list(dev_list)


# Create a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'dev': dev_dataset,
})

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["example"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=1024 #Changed fr0m 1024
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = dataset.map(tokenize_function, batched=True)
tokenizer.pad_token = tokenizer.eos_token


# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)


# output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'
# hyperparameters
lr = 1e-4
batch_size = 16
num_epochs = 3

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir="r32_inst_apollo_text_generation",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
    debug="underflow_overflow"
    
)



# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["dev"],
    args=training_args,
    data_collator=data_collator
)
print("----------------------- START TRAINING -----------------------")

# train model
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# with open('apollo_output_r32.txt', 'w') as f:
#     for i in range(50):
#         # Print to file instead of console
#         f.write(f"Prompt number: {i}\n\n")
        
#         comment = test['summary'][i]
#         true_answer = test['label'][i]

#         # Update your prompt here as needed
#         prompt = f'''<|system|>:\n {instructions_string} <|user|>: {comment} \n Would this patient get re-admitted to Intensive Care Unit?\n {output_string} \n<|assistant|>:\n '''
        
#         trainer.model.eval()
        
#         # Create the dash line (100 dashes)
#         dash_line = '-' * 100
        
#         # Tokenize the input prompt
#         inputs = tokenizer(prompt, return_tensors="pt")
        
#         # Generate the model's output
#         outputs = trainer.model.generate(
#             input_ids=inputs["input_ids"].to("cuda"), 
#             max_new_tokens=512,
#             temperature=0.3, 
#             do_sample=True
#         )
        
#         # Write the results to the file
#         f.write(f"{tokenizer.batch_decode(outputs)[0]}\n")
#         f.write(f"\nTrue Answer: {true_answer}\n\n")
#         f.write(f"{dash_line}\n")

#Saving the model locally
trainer.save_model("/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/Apollo/best_outputr32")

fine_tuned_model ="Apollo_unbalanced_r32"
trainer.model.save_pretrained(fine_tuned_model)
print("saved")


#Merging the model with Lora weights 
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage = True,
    return_dict = True,
    torch_dtype = torch.float16,
    device_map = {"": 0}
)


fine_tuned_merged_model = PeftModel.from_pretrained(base_model, fine_tuned_model)
fine_tuned_merged_model = fine_tuned_merged_model.merge_and_unload()


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
fine_tuned_merged_model.save_pretrained("Apollo_unbalanced_r32", safe_serialization = True)
tokenizer.save_pretrained("Apollo_unbalanced_r32")
tokenizer.padding_side = "right"

#Pushing the model to hugging face 
tokenizer.pad_token = tokenizer.eos_token
fine_tuned_merged_model.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)
trainer.model.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)
tokenizer.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)


