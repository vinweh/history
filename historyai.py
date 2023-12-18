
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
    client = openai.OpenAI()

    #openai.api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    print(f"Using model {model}")
    
   
    system_content = open("./system-prompt-csv.txt").read()
    
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
            # We've reached already reached our token limit
            print(f"Token limit reached at {index}")
            # let's first get an intermediate response before moving on with this message
            response = client.ChatCompletion.create(
                     model = model
                    ,temperature=0.6
                    ,messages=messages
                    ,response_format={"type" : "text"}
                )
            # grab the contents of that response and append it to the responses list
            r = response.choices[0].message['content'].strip()
            responses.append(r)
            num_tokens = num_tokens_for_message(message, model) # count from the last message that ddn't make it
            #reset for the the next batch of messages and append
            messages = []
            messages.append({"role": "system", "content": system_content})
            messages.append(message)
  
    # no more messages to append
    messages.append({"role": "user", "content" : "ALL SENT"})
    # get the last response (which is also the first if token limit was never reached)
    response = client.ChatCompletion.create(
                    model = model
                    ,temperature=0.6
                    ,messages=messages
                    ,response_format={"type" : "text"}
                )
 
    
    if not model_used:
        model_used = response.model
        print(f"Model actually used: {model_used}")
    # add the content         
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
    sep = ","
    header = ["\"Title\"","\"Url\"", "\"Category\"","\"Confidence Level\""]
    urls = get_urls(limit=50) #increase limit to comfort level
    if urls:
       # write predictions to csv file, do something smarter later
        predictions = classify(urls)
        
        with open("classifications.csv", "w") as f:
            f.write(sep.join(header))
            f.write("\n")    
            for idx, p in enumerate(predictions):
                f.write(p)
                
    else:
        print(f"No URLs to classify. Check log for errors.")

if __name__ == "__main__":
    sys.exit(main())