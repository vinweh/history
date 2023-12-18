import os
import sys
from contextlib import closing

from openai import OpenAI

from historydb import HistoryDb
from utils import num_tokens_for_message


def classify(history_items):
    """Very naively classify the history items (url/title pairs) and provide confidence
    """
    client = OpenAI()
    model = "gpt-3.5-turbo-1106"
    temperature = 0.3
    system_content = open("./system-prompt-csv.txt").read()
    messages = []
    responses = []
    num_tokens = 0
    max_tokens = 2000

    print(f"Rows found {len(history_items)}")
    sys_message = {"role": "system", "content": system_content}
    sys_tokens = num_tokens_for_message(sys_message, model)
    num_tokens += sys_tokens
    messages.append(sys_message)
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
            msg_count = len(messages)
            print(f"Sending {msg_count} messages to OpenAI...")
            response = client.chat.completions.create(
                     model = model
                    ,temperature=temperature
                    ,messages=messages
                    ,response_format={"type" : "text"}
                )
            
            # grab the contents of that response and append it to the responses list
            r = response.choices[0].message.content
            responses.append(r)
            num_tokens = num_tokens_for_message(message, model)
            # message queue reset for the the next batch of messages, reset sys message and append last message
            messages = []
            messages.append(sys_message)
            num_tokens += sys_tokens
            messages.append(message)
  
    # no more messages to append
    all_sent_message = ({"role": "user", "content" : "ALL SENT"})
    num_tokens += num_tokens_for_message(all_sent_message, model)
    messages.append(all_sent_message)
    # get the last response (which is also the first if token limit was never reached)
    msg_count = len(messages)
    print(f"Sending {msg_count} messages to OpenAI...")
    response = client.chat.completions.create(
                    model = model
                    ,temperature=temperature
                    ,messages=messages
                    ,response_format={"type" : "text"}
                )
    
    if not model_used:
        model_used = response.model
        print(f"Model actually used: {model_used}")
    
    # add the content         
    final_response = response.choices[0].message.content
    responses.append(final_response)   
    return responses

def get_urls(limit):
    """Get the URL, Title pairs from the history database"""
    
    browser_profile_path  = os.getenv("EDGE_BROWSER_PROFILE_PATH")
    history_db = os.path.join(os.environ['HOME'], browser_profile_path, "History")
    
    print(f"Attempt loading history DB {history_db}")
    h = HistoryDb(history_db)
    urls = h.get_urls(limit=limit)
    return urls

def write_to_csv(csv_file, predictions, sep=","):
    """"""
    sep = ","
    header = ["\"Title\"","\"Url\"", "\"Category\"","\"Confidence Level\""]
    with open(csv_file, "w") as f:
        f.write(sep.join(header))
        f.write("\n")    
        for idx, p in enumerate(predictions):
            if not p.endswith("\n"):
                p = "".join([p, "\n"])
            f.write(p)
       
def main():
    """The main train"""
    
    csv_file = "classifications.csv"
    urls = get_urls(limit=5) # decrease or increase limit to comfort level
    if urls:
       # write predictions to csv file, do something smarter later
        predictions = classify(urls)
        write_to_csv(csv_file, predictions)       
    else:
        print(f"No URLs to classify. Check log for errors.")

if __name__ == "__main__":
    sys.exit(main())