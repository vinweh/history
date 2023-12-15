
import os
import sys
import json
from contextlib import closing
import openai

from historydb import HistoryDb
from utils import num_tokens_for_message


def classify(history_items):
    """Very naively classify the history items (url/title pairs) and provide confidence
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    print(f"Using model {model}")
    
   
    system_content = r"""You are a content classifier. 
    Predict the most likely content category for each URL and TITLE pair provided. I'll prompt ALL SENT when done. 
    Do not attempt to access the URLs, just look at the URL text and TITLE text to classify.
    
    Structure your output in JSON as follows
    ===================================
    {  
        "Title" : "<TITLE>",
        "Url" : "<URL>",
        "Category" : "<CATEGORY>",
        "Confidence Level" : "<CONFIDENCE LEVEL>"
    }

    Separate each prediction using ","

    For example, I will provide you with the following information to classify: 

    TITLE: Top 10 La Liga goalscorers of all time
    URL: https://www.sportskeeda.com/football/la-liga-top-10-goalscorers-all-time

    and you will responds in JSON format as follows:

    {   "Title" : "Top 10 goals in La Liga in 2023",
        "Url" : "https://www.sportskeeda.com/football/la-liga-top-10-goalscorers-all-time",
        "Category" : "Sports",
        "Confidence Level" : "High" },
    """
    
    messages = []
    responses = []
    num_tokens = 0
    max_tokens = 2000

    print(f"Rows found {len(history_items)}")
    message = {"role": "system", "content": system_content}

    num_tokens += num_tokens_for_message(message, model)
    messages.append(message)
    model_used = None
    for index, row in enumerate(history_items):
        
         url, title = row
         content = f"URL: {url}, TITLE: {title}"
         message = {"role" : "user", "content" : content}
         num_tokens += num_tokens_for_message(message, model)

         if num_tokens <= max_tokens:
             messages.append(message)
             
         else:
            # We've reached token limit
            print(f"Token limit reached at {index}")
            
            response = openai.ChatCompletion.create(
                    model = model
                    ,temperature=0.6
                    ,messages=messages
                    ,response_format={"type" : "text"}
                )
            
            r = response.choices[0].message['content'].strip()
            responses.append(r)
            num_tokens = num_tokens_for_message(message, model) # count from the last msg
            messages = []
            messages.append(message)
  
    messages.append({"role": "user", "content" : "ALL SENT"})
    response = openai.ChatCompletion.create(
                    model = model
                    ,temperature=0.6
                    ,messages=messages
                    ,response_format={"type" : "text"}
                )
 
    
    if not model_used:
        model_used = response.model
        print(f"Model actually used: {model_used}")
            
    final_response = response.choices[0].message["content"].strip()
    responses.append(final_response)
    return responses

def get_urls(limit):
    """Get the URL, Tile pairs from the database"""
    
    browser_profile_path  = os.getenv("EDGE_BROWSER_PROFILE_PATH")
    history_db = os.path.join(os.environ['HOME'], browser_profile_path, "History")
    
    print(f"Attempt loading history DB {history_db}")
    h = HistoryDb(history_db)
    urls = h.get_urls(limit=limit)
    return urls

   
def main():

    urls = get_urls(limit=2) #increase limit to comfort level
    if urls:
       # write predictions to stdout and txt file, do something smarter later
        predictions = classify(urls)
        
        with open("classifications.json", "w") as f:
            f.write("{ \"Predictions\" : \n\t[\n")
            
            for idx, p in enumerate(predictions):
                #sys.stdout.write("\t\t%s\n" % p)
                f.write("\t\t%s\n" % p)
                
                    
                
            f.write("\t]\n}")    
    else:
        print(f"No URLs to classify. Check log for errors.")


if __name__ == "__main__":
    sys.exit(main())