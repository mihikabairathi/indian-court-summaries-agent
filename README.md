# indian-court-summaries-agent
A fine-tuned LLM available to legal practitioners (students and professionals alike) to generate high-quality summaries/headnotes of Indian legal court judgements. 

## Overview
With the exponential advancements in ML and NLP, as well as the ongoing digitization of legal documents around the world, Legal AI is an emerging space of opportunities - legal NER, case document summarization, case judgement prediction, and more. At the same time, Legal AI is tricky due to the lengthy documents and complex jargon used.

Legal AI could be incredibly useful in countries such as India, which has an overburdened legal system, and documents are often unstructured or lacking standard notation. And yet, most prominent Legal AI tools of today are geared towards the law of western countries. 

The sheer volume of judgements produced in India is quite high, and the length of these can often go beyond 100 pages each. As a result, law practitioners rely on legal experts and editors to write headnotes for court judgements. Headnotes are a quick guide for readers to understand the core legal issues, opinions of the judges, the laws cited and how they were interpreted, and the final decision. However, they take time and effort to write up. 

The LLMs of today could write up a headnote, but not in the tone and format that legal practitioners have come to expect. On the other hand, a fine-tuned model on pairs of judgements and already-written headnotes specifically of the Indian judicial system could lead to similarly toned/styled headnotes for unseen judgements. 

## Trained Model
Final model available on [Hugging Face](https://huggingface.co/mihikabairathi/merged-llama-indian-court-headnotes). Test dataset metrics:
- Rouge L Sum score: 0.495
- BERT f1 score: 0.86

It is a [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model fine-tuned on 7k judgement-headnote pairs. 

## Local Setup and Resources
- Install dependencies by running `pip install -r requirements.txt`
- If you want to run the fine tuning script for yourself, it is recommended to have around 40GB of VRAM. The NVIDIA A100 Tensor Core GPU available on Google Colab is ideal.
- If you want to simply run the inference script, it is recommended to have around 15GB of VRAM. Again, the best resources are on the NVIDIA A100 Tensor Core GPU for the fastest inference, but others can be used as well. To run inference on a text file, run `python3 scripts/generate.py --help`

## My Approach
### Step 1: Curate the dataset
- I used a dataset of 7k Indian Supreme Court Judgements and corresponding headnotes. This data set had already been scraped and made available for use [here](https://github.com/Law-AI/summarization). 

### Step 2: Filter the dataset
- I went over the dataset and found that the quality was a bit inconsistent. A big factor in successfully fine-tuning a model is the quality of the dataset used. I wanted to filter out poor headnotes, and oversample the high-quality headnotes. 
- I came up with a basic rubric that highlighted the key components and requirements of a good headnote. Using in-context learning methods, I had an open-source LLM output True/False assignments to rubric components for each headnote, and then with some post-processing I assigned each headnote a label - poor, medium, or great. 

### Step 3: Fine-Tuning
- After filtering and oversampling headnotes in the dataset, I worked with various open-source LLMs and parameters and fine-tuned different models using the PEFT technique QLoRA.
- I evaluated model performances using cross-entropy loss, Rouge, and BERT scores, before picking a final model. 

### Step 4: Inference
- After picking the best model configuration and generation technique, I merged the LoRA weights with the original model, and uploaded a final model to Hugging Face. 
- I wrote a script that can run the model anywhere by downloading it from Hugging Face.

## Future Possibilities
- Deploy online for easy usage, using persistent GPU resources. 
- Reduce the initial context size by generating extractive summaries, and then using that to create the abstractive summary (headnote). This could help improve performance, resource usage, and inference time. 
- Curate a dataset to train a rewards model for RLHF. This could help improve perforance. 

## Authors
[Mihika Bairathi](https://www.linkedin.com/in/mihikabairathi/)

## Citations
The original dataset was obtained from the following papers:

1. **Shukla, A.**, **Bhattacharya, P.**, **Poddar, S.**, **Mukherjee, R.**, **Ghosh, K.**, **Goyal, P.**, & **Ghosh, S.** (2022).  
   *Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation*.  
   In *The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing*.

   <details>
     <summary><code>BibTeX</code></summary>

   ```bibtex
    @inproceedings{shukla2022,
        title={Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation},
        author={Shukla, Abhay and Bhattacharya, Paheli and Poddar, Soham and Mukherjee, Rajdeep and Ghosh, Kripabandhu and Goyal, Pawan and Ghosh, Saptarshi},
        booktitle={The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing},
        year={2022}
    }
    ```

2. **Bhattacharya, P.**, **Hiware, K.**, **Rajgaria, S.**, **Pochhi, N.**, **Ghosh, K.**, & **Ghosh, S.** (2019).  
   *A Comparative Study of Summarization Algorithms Applied to Legal Case Judgments*.  
   In *European Conference on Information Retrieval* (pp. 413–428). Springer.

   <details>
     <summary><code>BibTeX</code></summary>

   ```bibtex
    @inproceedings{bhattacharya2019comparative,
        title={A comparative study of summarization algorithms applied to legal case judgments},
        author={Bhattacharya, Paheli and Hiware, Kaustubh and Rajgaria, Subham and Pochhi, Nilay and Ghosh, Kripabandhu and Ghosh, Saptarshi},
        booktitle={European Conference on Information Retrieval},
        pages={413--428},
        year={2019},
        organization={Springer}
    }
    ```

## Acknowledgements
This project was created during a batch at the [Recurse Center](https://www.recurse.com/), and fellow recursers there gave a lot of advice, feedback, and support. 