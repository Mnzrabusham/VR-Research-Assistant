"""
Testing fine-tuned model
"""

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read fine-tuned model ID
try:
    with open("finetuned_model_id.txt", "r") as f:
        model_id = f.read().strip()
except FileNotFoundError:
    print("Error: finetuned_model_id.txt not found")
    print("Run finetune_openai.py first and wait for completion")
    exit(1)

print(f"Testing fine-tuned model: {model_id}\n")

# Test questions
test_questions = [
    "What is world-space cueing?",
    "How does eye tracking work in VR?",
    "What are the main challenges in VR attention guidance?",
    "How effective is gaze guidance for improving VR experiences?"
]

for question in test_questions:
    print(f"Question: {question}")
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert VR research assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    print(f"Answer: {answer}\n")
    print("-" * 80 + "\n")

print("Testing complete!")

