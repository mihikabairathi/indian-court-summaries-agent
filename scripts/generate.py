from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import nltk
import re
import torch

def load_model_and_tokenizer(dtype):
    model = AutoModelForCausalLM.from_pretrained(
        "mihikabairathi/merged-llama-indian-court-headnotes", 
        torch_dtype=dtype, 
        device_map="auto",
        # for QLoRA
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained("mihikabairathi/merged-llama-indian-court-headnotes")
    return model, tokenizer

def generate_instructions():
    return """\
Case judgements represent rulings issued by judges in judicial proceedings, often comprising extensive transcripts spanning dozens of pages.
Your task is to compose the headnote for the given case judgement. An excellent headnote captures the core of the judgment properly, succinctly, and completely.

Here are the core elements of a headnote:
- PAST TENSE: The headnote is written in the past tense.
- METADATA: The headnote includes the case name, judgement number, court, judge(s), and date of judgement.
- INTRODUCTION: The headnote introduction describes the field of law that the case deals with, and does not directly jump into the case details.
- KEY FACTS: The headnote includes who filed the case, why the case was filed, and what remedy the filer wants.
- KEY ARGUMENTS: The headnote includes both sides' submissions and reasons to support their position.
- RELEVANT LAWS: The headnote references relevant legislation and concepts to support the case judgement.
- CONCLUSION: The headnote includes the case's conclusion and procedural disposition (ex: permitted, dismissed, reversed, remanded, affirmed, etc.).

VERY IMPORTANT: Do not generate multiple paragraphs or sections. Write a single paragraph that does not exceed 800 words, or a single page.
"""

def tokenize_input(input_text, tokenizer):
    instruction = generate_instructions()
    prompt = f"""\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return tokenizer(prompt, return_tensors='pt')

def clean_headnote(headnote, correct_words):
    # remove prompt and extra headings, as eos token doesn't always work
    headnote = headnote.split('### Response:')[1].split('#')[0]

    # remove last sentence if its not complete - this is a naive approach but works in most cases
    last_seen_period = headnote.rfind('.')
    if last_seen_period != len(headnote) - 1:
        headnote = headnote[:last_seen_period+1]

    # double spacing issues
    headnote = re.sub(r'\s{2,}', ' ', headnote)

    # handle new lines - if there is a \n but it does not precede a Capital letter and does not succeed a period then replace with a space
    headnote = re.sub(r'(?<![^0-9]\.)\n|(?<=\.)\n(?![A-Z])', ' ', headnote)

    # random HELD words
    headnote = re.sub('HELD', 'held', headnote)
    headnote = re.sub('^HELD', 'held', headnote)

    # fix spelling mistakes
    headnote_words = headnote.split()
    for i in range(len(headnote_words) - 1):
        left_word = headnote_words[i]
        right_word = headnote_words[i+1]
        if left_word.lower() not in correct_words and right_word.lower() not in correct_words and left_word.lower()+right_word.lower() in correct_words:
            headnote = re.sub(f'{left_word} {right_word}', f'{left_word}{right_word}', headnote)

    # last-minute cleaning
    return headnote.strip()

def model_call(tokenized_input, model, tokenizer, correct_words):
    outputs = model.generate(
        input_ids=tokenized_input['input_ids'].to("cuda"),
        attention_mask=tokenized_input['attention_mask'].to("cuda"),
        max_new_tokens=1000,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('###')],
        repetition_penalty=1.5,
        no_repeat_ngram_size=4,
        num_beams=4,
        length_penalty=1.2
    )

    generated_headnote = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    generated_headnote = clean_headnote(generated_headnote, correct_words)
    return generated_headnote

def generate_headnote(input_text, dtype):
    model, tokenizer = load_model_and_tokenizer(dtype)

    nltk.download('words')
    correct_words = nltk.corpus.words.words()

    tokenized_input = tokenize_input(input_text, tokenizer)
    generated_headnote = model_call(tokenized_input, model, tokenizer, correct_words)
    return {"headnote": generated_headnote}

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fine-tuned model with specified dtype on the given input text.")
    parser.add_argument("input_file", help="Path to the input text file containing the case judgement.")
    parser.add_argument("--use_bf16", action="store_true", help="Use torch.bfloat16 instead of float16")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    input_text = read_file(args.input_file)
    print(generate_headnote(input_text, dtype))