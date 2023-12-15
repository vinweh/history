import os
import requests
import json
import openai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_content_categories(urls):
    
    endpoint = "https://api.openai.com/v1/classifications"

    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "text-content-classification",
        "examples": [{"text": url, "label": title} for url, title in urls]
    }

    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code == 200:
        predictions = response.json()
        return predictions
    else:
        print(f"Failed with status code {response.status_code}")
        return None

# List of URLs and their titles
urls_and_titles = [
    ("https://www.msn.com/en-us/news/us/apology-letters-by-sidney-powell-and-kenneth-chesebro-in-georgia-election-case-are-one-sentence-long/ar-AA1lwldl"
     , "Apology letters by Sidney Powell and Kenneth Chesebro in Georgia election case are one sentence long"),
    ("https://example2.com", "Example 2 Title"),
    # Add more URLs and titles as needed
]

# Get content categories for URLs
predictions = get_content_categories(urls_and_titles)

if predictions:
    for idx, prediction in enumerate(predictions["classifications"]):
        url, title = urls_and_titles[idx]
        print(f"URL: {url}, Title: {title}")
        print(f"Predicted category: {prediction['label']}")
        print("Confidence score:", prediction['confidence'])
        print("--------------------")
