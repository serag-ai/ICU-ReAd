# Leveraging Large Language Models to Predict Unplanned ICU Readmissions from Electronic Health Records


[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)


---

## Overview

This repository contains the code and models introduced in our paper:  
> **"Leveraging large language models to predict unplanned ICU readmissions from electronic health records"**  
> *Helmy, Hoda, Ahmed Ibrahim, Maryam Arabi, Aamenah Sattar, and Ahmed Serag. Natural Language Processing Journal (2025)*
> *[Paper](https://www.sciencedirect.com/science/article/pii/S2949719125000585)*

This project explores the use of large language models (LLMs) to predict ICU readmissions and generate clinical explanations. Two open-source models, Gemma 2B and Apollo 2B, were fine-tuned and compared.

---

## Key Features

- Utilized Large Language Models (LLMs) to predict unplanned ICU re-admission from Electronic Health Records (EHRs).
- Developed a serialization approach to transform structured EHR data into a text-based format for LLM processing. 
- Investigated explicit classification (binary labels) and implicit classification (text generation) to enhance interpretability.  
- Demonstrated the potential of LLMs in clinical decision support by generating interpretable insights for ICU physicians.

---
## Dataset
https://eicu-crd.mit.edu/gettingstarted/access/
---

## Model from HUGGING FACE 🤗

| Model Name | Parameters | Description | Hugging Face Link |
|-------------|-------------|--------------|-------------------|
| Explicit Gemma 2B (FT)| 3B | Explicit Classification method| [🤗 View Model](https://huggingface.co/serag-ai/ICU-GEMMA-EXP) |
| Explicit Apollo 2B (FT) | 3B | Explicit Classification method | [🤗 View Model](https://huggingface.co/serag-ai/ICU-APOLLO-EXP) |
| Implicit Gemma 2B (FT) | 3B | Implicit Classification method with text generation | [🤗 View Model](https://huggingface.co/serag-ai/ICU-GEMMA-IMP) |
| Implicit Apollo 2B (FT)| 3B | Implicit Classification method with text generation | [🤗 View Model](https://huggingface.co/serag-ai/ICU-APOLLO-IMP) |

> To load models directly in Python for text generation:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "HodaHelmy/gemma_r32_unbalanced" 
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
