import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore", category=FutureWarning)

tokenizer = AutoTokenizer.from_pretrained(
    "Vamsi/T5_Paraphrase_Paws", legacy=True)
paraphrasing_model = AutoModelForSeq2SeqLM.from_pretrained(
    "Vamsi/T5_Paraphrase_Paws")


def para(sentence):
    text = "paraphrase: " + sentence + " </s>"
    # </s> to indicate end of a sequence
    encoding = tokenizer.encode_plus(
        text, padding='max_length', max_length=256, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding[
        "attention_mask"  # tokenize text
    ]

    outputs = paraphrasing_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=256,
        do_sample=True,  # use sampling during generation
        top_k=120,  # number of highest probability tokens to consider
        top_p=0.95,  # cumulative probability threshold for sampling
        num_return_sequences=1,
    )
    # decodes the generated output
    line = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return line


para("hello world")
