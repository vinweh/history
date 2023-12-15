
import os
import sys
import logging
from contextlib import closing
import openai

from historydb import HistoryDb
from utils import num_tokens_for_message


log_file = "./log.txt"
format = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.DEBUG, filename=log_file,
                    format=format,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



def classify(history_items):
    """Very naively classify the history items (url/title pairs) and provide confidence
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    logging.info("Using model %s", model)
    
   
    system_content = r"""You are a content classifier. 
    Predict the most likely content category for each URL and TITLE pair provided. I'll prompt ALL SENT when done. 
    Do not attempt to access the URLs, just look at the URL text and TITLE text to classify.
    
    Structure your output as follows:
    ===================================
    Title: <TITLE>
    Url: <URL>
    Category: <CATEGORY>
    Confidence Level: <CONFIDENCE LEVEL>

    For example, I will provide you with the following information to classify: 

    TITLE: Top 10 La Liga goalscorers of all time
    URL: https://www.sportskeeda.com/football/la-liga-top-10-goalscorers-all-time

    and you will responds as follows:

    Title: Top 10 goals in La Liga in 2023
    Url: https://www.sportskeeda.com/football/la-liga-top-10-goalscorers-all-time
    Category: Sports
    Confidence Level: High
    """

    messages = []
    responses = []
    num_tokens = 0
    max_tokens = 2000

    logging.info("Rows found %i", len(history_items))
    message = {"role": "system", "content": system_content}

    num_tokens += num_tokens_for_message(message, model)
    messages.append(message)
    
    for index, row in enumerate(history_items):
        
         content = "URL: %s, TITLE: %s" % row
         message = {"role" : "user", "content" : content}
         num_tokens += num_tokens_for_message(message, model)

         if num_tokens <= max_tokens:
             messages.append(message)
             logging.info("Added row %i", index)
         else:
            logging.info("Token limit reached at %i" , index)
            
            response = openai.ChatCompletion.create(
                    model = model
                    ,temperature=0.6
                    ,messages=messages 
                )
            
            r = response.choices[0].message['content'].strip()
            responses.append(r)
            num_tokens = num_tokens_for_message(message, model) # count from the last msg
            messages = []
            messages.append(message)
            logging.info("Added row %i", index)


    messages.append({"role": "user", "content" : "ALL SENT"})
    response = openai.ChatCompletion.create(model=model, messages=messages)
    final_response = response.choices[0].message["content"].strip()
    responses.append(final_response)
    return responses

def main():

    browser_profile_path  = os.getenv("EDGE_BROWSER_PROFILE_PATH")
    history_db = os.path.join(os.environ['HOME'], browser_profile_path, "History")
    logging.info("Loading history DB: %s", history_db)
    h = HistoryDb(history_db)
    h.load(limit=20)

    predictions = classify(h.rows)
    # write to stdout
    for p in predictions:
        sys.stdout.write("%s\n" % p)


if __name__ == "__main__":
    sys.exit(main())