#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import pandas as pd
import os
import time
import datetime
import gc
import random
import wandb
from nltk.corpus import stopwords
import re
import torch
import transformers
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
from evaluate import evaluator
import evaluate
from scipy.special import softmax
from datasets import (
    Dataset,
    DatasetDict)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from torch.utils.data import (
    Dataset)
from peft import PeftModel
from json import encoder
import csv
from sklearn.preprocessing import LabelEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

w_token ="YOUR_TOKEN"
r_token ="YOUR_TOKEN"



#LOADING THE FILE
def load_file(data_file):
    df = pd.read_csv(data_file)
    print('File loaded successfully')
    return df

# Load the CSV file
file_path_2 = '/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/llm.csv'
df2 = load_file(file_path_2)


# In[4]:


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


# In[5]:


prompts, answers = create_prompts(df2)


# In[6]:


encoder = LabelEncoder()
labels = answers
encoded_labels = encoder.fit_transform(labels)


# In[9]:


# Stratified splitting of the dataset into train, dev, and test with a 60:20:20 ratio
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

train.to_csv("train_set_r32_seqGem_25.csv")
dev.to_csv("dev_set_r32_seqGem_25.csv")
test.to_csv("test_set_r32_seqGem_25.csv")

label_encoder = LabelEncoder()
all_labels = pd.concat([train['label'], dev['label'], test['label']])
label_encoder.fit(all_labels)

# Apply the label encoding to the 'LABEL ' column
for df in [train, dev, test]:
    df['label_Encoded'] = label_encoder.transform(df['label'])


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
dataset


# In[12]:


class_weights = (1/train.label_Encoded.value_counts(normalize=True).sort_index()).tolist()
class_weights = torch.tensor(class_weights)
class_weights = class_weights/class_weights.sum()
class_weights


# In[13]:


commit_message = "Fine-tuning google/gemma-2b-it on ICU dataset with shorter prompt"

model_name = "google/gemma-2b-it"

wandb.login(key="30b0271bbe141a329715cc74ff244a878c770e8c")


# In[14]:


import numpy as np
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
num_labels = len(np.unique(encoded_labels))
print(num_labels)

#"TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForSequenceClassification.from_pretrained(model_name, #model_name
                                             quantization_config=bnb_config,
                                             device_map="auto",# automatically figures out how to best use CPU + GPU for loading model
                                             token=r_token,
                                             num_labels=num_labels,
                                             trust_remote_code=False) # prevents running custom model files on your machine
                                             #revision="main") # which version of model to use in repo

tokenizer = AutoTokenizer.from_pretrained(model_name, #model_name
                                        token=r_token, add_eos_token=True, use_fast=True)

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token


# In[15]:


model.train() # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)


# LoRA config
config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj","v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

# LoRA trainable version of model
model = get_peft_model(model, config)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1
# trainable parameter count
# print_trainable_parameters(model)
model.print_trainable_parameters()


# In[16]:


sentences = test.summary.tolist()



all_outputs = []
batch_size = 32  # Example batch size

# Loop through sentences in batches
for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]

    # Tokenize the batch of sentences
    inputs = tokenizer(batch_sentences, return_tensors="pt", 
    padding=True, truncation=True, max_length=512)

    # Move inputs to the appropriate device (GPU or CPU)
    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

    with torch.no_grad():
        # Get the model's outputs (logits)
        outputs = model(**inputs)
        all_outputs.append(outputs['logits'])

        
# Concatenate the logits from all batches
final_outputs = torch.cat(all_outputs, dim=0)

# Apply softmax to get probabilities
y_proba = torch.softmax(final_outputs, dim=1).cpu().numpy()

# Store predictions (class with highest probability) and probabilities in the dataframe
test['predictions'] = y_proba.argmax(axis=1)  # Class with the highest probability
test['prediction_probabilities'] = y_proba[:, 1]    # Predicted probabilities for each class


# In[17]:


def get_metrics_result(test_df):
    y_test = test_df.label_Encoded
    y_pred = test_df.predictions
    y_pred_proba = test_df.prediction_probabilities  # Now containing probability scores

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Assuming this is a binary classification problem
    if len(np.unique(y_test)) == 2:
        # Use probability scores for the positive class (usually the second column)
        print("AUC Score:", roc_auc_score(y_test, y_pred_proba))  
    else:
        # For multiclass classification, use one-vs-rest (OVR) strategy
        print("AUC Score:", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))

# Call the function to get metrics
get_metrics_result(test)


# In[18]:


def tokenize_function(examples):
    # extract text
    text = examples["summary"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=1024
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = dataset.map(tokenize_function, batched=True)
tokenized_data = tokenized_data.rename_column("label", "admission")
tokenized_data = tokenized_data.rename_column("label_Encoded", "label")
tokenizer.pad_token = tokenizer.eos_token
tokenized_data.set_format("torch")

# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)


# In[19]:


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights,
            dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


# In[20]:


lr = 1e-4
batch_size = 16
num_epochs = 2

#old
# lr = 1e-4
# batch_size = 8
# num_epochs = 3

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir='gemma2b_unbalanced_sequence_classification_r32_25',
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.05,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",

)

def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Compute precision, recall, f1, accuracy
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    
    # Compute AUC
    probabilities = softmax(logits, axis=-1)  # Convert logits to probabilities
    if len(np.unique(labels)) == 2:  # Binary classification
        auc = roc_auc_score(labels, probabilities[:, 1])
    else:  # Multiclass classification
        auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')
    
    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": accuracy,
        "auc": auc
    }


# In[21]:


trainer = CustomTrainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["dev"],
    args=training_args,
    compute_metrics=compute_metrics,
    data_collator = collate_fn,
    class_weights=class_weights
)
print("----------------------- START TRAINING -----------------------")
start_time = time.time()

# train model
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


results = trainer.evaluate(eval_dataset=tokenized_data["test"])
print("Test Metrics: ",results)


# In[22]:


def generate_predictions(model, test):
    sentences = test.summary.tolist()
    batch_size = 32  
    all_outputs = []

    for i in range(0, len(sentences), batch_size):

        batch_sentences = sentences[i:i + batch_size]

        inputs = tokenizer(batch_sentences, return_tensors="pt", 
        padding=True, truncation=True, max_length=512)

        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') 
        for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
        
    final_outputs = torch.cat(all_outputs, dim=0)
    y_proba = torch.softmax(final_outputs, dim=1).cpu().numpy()
    test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
    test['prediction_probabilities'] = y_proba[:,1]

generate_predictions(model,test)
get_metrics_result(test)


# In[30]:


def get_original_label(prediction):
    return label_encoder.inverse_transform([prediction])[0]
 
# Example of how to use the model for prediction
def predict_readmission(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = trainer.model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return get_original_label(predicted_class)
 
# Example usage
prediction_two = []
for i in range (len(test)):
    example_input = test['summary'][i]
    predictions = predict_readmission(example_input)
    prediction_two.append(predictions)

df = pd.DataFrame({
    "True Answer": test['label'],  # Assuming this is a list of true answers
    "Predicted": prediction_two,          # Assuming `predictions` is a list of predictions
})

df.to_csv('unbalanced_output_answers_finetuned_classification_r32_25.csv')


# In[24]:


trainer.save_model("/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/gemma_sequence/best_r32_output_25")


# In[25]:


fine_tuned_model ="Gemma2B_Seq_class"
trainer.model.save_pretrained(fine_tuned_model)
print("saved")


# In[26]:


base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    low_cpu_mem_usage = True,
    return_dict = True,
    torch_dtype = torch.float16,
    device_map = {"": 0}
)


# In[27]:


fine_tuned_merged_model = PeftModel.from_pretrained(base_model, fine_tuned_model)
fine_tuned_merged_model = fine_tuned_merged_model.merge_and_unload()


# In[28]:


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
fine_tuned_merged_model.save_pretrained("Gemma2B_Seq_class", safe_serialization = True)
tokenizer.save_pretrained("Gemma2B_Seq_class")
tokenizer.padding_side = "right"


# In[29]:


tokenizer.pad_token = tokenizer.eos_token
fine_tuned_merged_model.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)
trainer.model.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)
tokenizer.push_to_hub(fine_tuned_model, use_temp_dir=False,token=w_token)


# In[ ]:




