import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig

# Path to your fine-tuned LoRA adapter model
peft_model_path = "./finetuned_model"

# Base LLaMA model
base_model_name = "meta-llama/Llama-2-7b-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, peft_model_path)
model.eval()

# Inference function
def generate_response(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test set of example tweets
example_tweets = {
    "Positive": [
        "Apple shares soar after record-breaking iPhone sales this quarter 📈🍏",
        "Tesla reports better-than-expected earnings, stock jumps 12% 🚀",
        "Microsoft's cloud business continues to grow, pushing stock to all-time high ☁️📊"
    ],
    "Negative": [
        "Meta plunges after weak ad revenue and gloomy Q4 forecast 📉",
        "Google faces antitrust lawsuit; investors brace for a long battle ⚖️😬",
        "Intel stock tumbles as chip demand slows down and margins shrink 🧊"
    ],
    "Neutral": [
        "Amazon to report earnings tomorrow, analysts expect mixed results 🤔",
        "JP Morgan to hold annual shareholder meeting next week 🏛️",
        "Boeing resumes production at South Carolina facility following brief halt 🔧"
    ]
}

# Run inference on all tweets
if __name__ == "__main__":
    instruction = "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."

    for sentiment, tweets in example_tweets.items():
        print(f"\n=== {sentiment.upper()} TWEETS ===")
        for tweet in tweets:
            prompt = f"Instruction: {instruction}\nInput: {tweet}\nAnswer:"
            response = generate_response(prompt)
            print(f"\n📝 Tweet: {tweet}")
            print(f"📌 Model Response: {response.strip()}")
