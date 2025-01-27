#!/usr/bin/env python
# coding: utf-8


import torch
import pandas as pd
import os
import time
import datetime
import gc
import random
# from nltk.corpus import stopwords
import re
import transformers
import torch.nn as nn
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
# from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import evaluate 
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    random_split,
    Dataset)

#from trl import SFTTrainer
from peft import PeftModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

w_token ="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"
r_token ="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"


# In[2]:


# from huggingface_hub import login



commit_message = "Fine-tuning google/gemma-2b-it on ICU dataset with shorter prompt"

model_name = "google/gemma-2b-it"



# In[4]:


#commit_message = "Fine-tuning llama2-chat-7b on ICU dataset with shorter prompt"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#"TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name, #model_name
                                             quantization_config=bnb_config,
                                             device_map="auto",# automatically figures out how to best use CPU + GPU for loading model
                                             token=r_token,
                                             trust_remote_code=False) # prevents running custom model files on your machine
                                             #revision="main") # which version of model to use in repo

tokenizer = AutoTokenizer.from_pretrained(model_name, #model_name
                                        token=r_token, add_eos_token=True, use_fast=True)


# In[5]:


tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token



# In[6]:


def load_file(data_file):
    df = pd.read_csv(data_file)
    print('File loaded successfully')
    return df

# Load the CSV file
file_path_2 = '/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/llm.csv'
df2 = load_file(file_path_2)


# In[7]:


def filter_unwanted_words(diseases, unwanted_words):
    return [disease for disease in diseases if not any(word in disease for word in unwanted_words)]

# Function to create prompts for each row
def create_prompts(df):
    prompts = []
    admission_responses = []
    
    unwanted_words = ['performed']  # Add any other unwanted words here
    
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
        # Admission Type: {admission_type}
        # Response: {admission_response}

        # Add the comment to the list
        prompts.append(comment.strip())
        admission_responses.append(admission_response.strip())

    return prompts, admission_responses


# In[8]:


prompts, answers = create_prompts(df2)


# In[9]:


print(prompts[300])
print(answers[300])


# In[9]:


print(prompts[14])
print( answers[14])

model.eval() # model in evaluation mode (dropout modules are deactivated)
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
input = prompts[14]

prompt =f'''<start_of_turn>user\n {instructions_string} {input} \n Would this patient get re-admitted to Intensive Care Unit?<end_of_turn>\n {output_string} \n <start_of_turn>model\n '''
# generate output
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=1024, temperature=0.1, do_sample = True)
#print("without instruction")
print('------------------------------------------------')
print(tokenizer.batch_decode(outputs)[0])


# In[10]:


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

# In[12]:


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

train.to_csv("train_set_r32.csv")
dev.to_csv("dev_set_r32.csv")
test.to_csv("test_set_r32.csv")


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


# In[14]:


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


# In[15]:


from datasets import Dataset, DatasetDict
train_list = train.to_dict(orient='records')
dev_list = dev.to_dict(orient='records')
dataset = DatasetDict({
    'train': Dataset.from_list(train_list),
    'dev': Dataset.from_list(dev_list)
})
dataset


# In[16]:

example_template = lambda text, label: {
    'example': f'''<start_of_turn>user \n {instructions_string} \n{text} Based on his admission diagnoses, will this patient get readmitted to Intensive Care Unit?\n<end_of_turn>\n {output_string} \n <start_of_turn>model\n ''' + "\n" + label 
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


# In[17]:


# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)


# In[22]:


# hyperparameters
lr = 1e-4
batch_size = 16
num_epochs = 4

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir="r32_inst_gemma_text_generation",
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
)


# In[21]:


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

trainer.save_model("/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/Gemma_August/best_output_r32")


# In[35]:


fine_tuned_model ="gemma_r32_unbalanced"
trainer.model.save_pretrained(fine_tuned_model)
print("saved")


# In[36]:


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage = True,
    return_dict = True,
    torch_dtype = torch.float16,
    device_map = {"": 0}
)


# In[37]:


fine_tuned_merged_model = PeftModel.from_pretrained(base_model, fine_tuned_model)
fine_tuned_merged_model = fine_tuned_merged_model.merge_and_unload()


# In[38]:


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
fine_tuned_merged_model.save_pretrained("gemma_r32_unbalanced", safe_serialization = True)
tokenizer.save_pretrained("gemma_r32_unbalanced")
tokenizer.padding_side = "right"


# In[39]:


tokenizer.pad_token = tokenizer.eos_token
fine_tuned_merged_model.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)
trainer.model.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)
tokenizer.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)


