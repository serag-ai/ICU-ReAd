#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import os
import time
import datetime
import gc
import random
from nltk.corpus import stopwords
import re
import transformers
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import evaluate 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel, PeftConfig
import re
import nltk
from nltk import sent_tokenize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

w_token ="YOUR_TOKEN"
r_token ="YOUR_TOKEN"




#LOADING THE MODEL FROM HUGGING FACE
peft_model_id = "HodaHelmy/gemma_r32_unbalanced" #HUGGING FACE LIBRARY
config = PeftConfig.from_pretrained(peft_model_id)


model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)



dataset_path= '/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/Gemma_August/test_set_r32.csv' #UPLOAD TEST SET
test = pd.read_csv(dataset_path)
test = test.drop(columns=["Unnamed: 0"])


#INSTRUCTION STRING NEEDED FOR THE PROMPT 
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

# nltk.download('punkt')

# Initialize lists to store results
predicted_labels = []
predicted_probs = []


# Define your keywords for the search
yes_keywords = ["yes", "likely","possibility", "potential"]
no_keywords = ["no", "unlikely"]

# Define additional negations or contradictory patterns to watch for
negations = ["not", "unlikely", "no evidence"]

#FUNCTION TO EXTRACT KEYWORDS IN THE GENERATED OUTPUT
def analyze_sentences(sentences):
    yes_count = no_count = 0
    for sentence in sentences:
        # Check for negations within a sentence
        negated = any(neg in sentence for neg in negations)
        
        # Search for Yes/No keywords only if not negated or negated appropriately
        for keyword in yes_keywords:
            if keyword in sentence and not negated:
                yes_count += 1
                break
        for keyword in no_keywords:
            if keyword in sentence and negated:
                no_count += 1
                break
                
        # Handle negated cases 
        if any(neg in sentence for neg in negations):
            for keyword in yes_keywords:
                if keyword in sentence:
                    no_count += 1  # Treat "Yes" under negation as "No"
                    break
            for keyword in no_keywords:
                if keyword in sentence:
                    yes_count += 1  # Treat "No" under negation as "Yes"
                    break

    return yes_count, no_count

# Placeholder for vocab check
vocab = tokenizer.get_vocab() if tokenizer else {}

#LOOP TO GENERATE OUTPUT FROM THE FULL DATASET
with open('generated_output_evaluation_r32_temp0.3.txt', 'w') as f:
    for i in range(len(test)): #YOU CAN ADD THE RANGE OF EXAMPLE YOU WANT HERE WE USED THE FULL DATASET (CONSUMES TIME)
        f.write(f"Prompt number: {i}\n\n")
        print(f"Processing Prompt {i}...")

        # Retrieve the prompt and true answer
        comment = test.iloc[i]['summary']
        true_answer = test.iloc[i]['label']

        # Construct the prompt for the model
        prompt = (
            f"<start_of_turn>user\n {instructions_string} {comment} \n"
            f"Would this patient get re-admitted to Intensive Care Unit?<end_of_turn>\n"
            f"{output_string} \n<start_of_turn>model\n"
        )
        # print(prompt)
        model.eval()
        model.to("cuda")

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        # Generate the model's output
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"].to("cuda"),
                attention_mask=inputs['attention_mask'].to("cuda"),
                renormalize_logits=False,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode the generated output
        generated_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()

        # Extract text after "model" and lower case
        generated_text = generated_output.lower().split('model', 1)[-1].strip()
        

        # Tokenize into sentences and analyze
        sentences = sent_tokenize(generated_text)
        print("generated_text:", sentences )
        yes_count, no_count = analyze_sentences(sentences)
        print("Yes Count: ", yes_count,' No Count: ', no_count  )

        # Determine predicted label
        if yes_count > no_count:
            predicted_label = 1
        elif no_count > yes_count:
            predicted_label = 0
        else:
            predicted_label = 0 if no_count > 0 else 1  # Default to "Yes" on tie
        
        print("predict: ", predicted_label)

        predicted_labels.append(predicted_label)

        outputs_log = model(**inputs.to("cuda"))

        # logits from the model output
        logits = outputs_log.logits  # Shape: [batch_size, sequence_length, vocab_size]

        if logits is not None:
            # the last token's logits for probabilities
            last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            probabilities = F.softmax(last_logits, dim=-1).cpu().numpy()  # Normalize logits

            # Get token IDs for "yes" and "no"
            yes_token_id = tokenizer.convert_tokens_to_ids("yes")
            no_token_id = tokenizer.convert_tokens_to_ids("no")

            # Extract probabilities
            yes_prob = probabilities[0][yes_token_id] if yes_token_id < probabilities.shape[1] else 0
            no_prob = probabilities[0][no_token_id] if no_token_id < probabilities.shape[1] else 0

            # print("yes token:", probabilities[0][yes_token_id] if yes_token_id < probabilities.shape[1] else "Out of range")
            # print("yes_prob:", yes_prob)
            # print("no token:", probabilities[0][no_token_id] if no_token_id < probabilities.shape[1] else "Out of range")
            # print("no_prob:", no_prob)

            predicted_probs.append([yes_prob, no_prob])

        # SAVING GENERATED output to file
        f.write(f"Generated Output:\n{generated_output}\n")
        f.write(f"True Answer: {true_answer}\n\n")
        f.write('-' * 100 + '\n')


# Convert test labels to binary (1 for 'yes', 0 for 'no')
test_labels = test['label']
test_labels = [1 if 'yes' in label.lower() else 0 for label in test_labels]

# Ensure predicted_labels and predicted_probs have valid lengths
if len(predicted_labels) != len(test_labels) or len(predicted_probs) != len(test_labels):
    raise ValueError("Mismatch in lengths of predictions, probabilities, or test labels.")

# Create a DataFrame from the collected data
df = pd.DataFrame({
    "True Answer": test_labels,
    "Predicted": predicted_labels,
    "Predicted Probability Yes": [p[0] for p in predicted_probs],
    "Predicted Probability No": [p[1] for p in predicted_probs],
})

# Save predictions to CSV
df.to_csv('unbalanced_test_generated_output_evaluation_gemma2br32.csv', index=False)

# Filter out invalid predictions (-1, if they exist)
valid_df = df[df["Predicted"] != -1]

# Ensure there's valid data before evaluation
if valid_df.empty:
    print("No valid predictions for evaluation.")
else:
    # Calculate accuracy, F1 score, and AUC
    accuracy = accuracy_score(valid_df["True Answer"], valid_df["Predicted"])
    f1 = f1_score(valid_df["True Answer"], valid_df["Predicted"], average='weighted')
    roc_auc = roc_auc_score(valid_df["True Answer"], valid_df["Predicted Probability Yes"])

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")


##### CLASSIFICATION REPORT 
print("Classification Report:")
print(classification_report(valid_df["True Answer"], valid_df["Predicted"]))



####### CONFUSION MATRIX 

cm = confusion_matrix(valid_df["True Answer"], valid_df["Predicted"])

print(cm)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels =['One Admission','Re-admission'])
cm_display.plot(cmap ='GnBu')
plt.title('Gemma 2B confusion matrix')
plt.savefig('gemma2b_confusion_matrix_r32_whole_test.png', dpi=300, bbox_inches='tight')
# plt.show




