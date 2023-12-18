import os
import sys

from historydb import HistoryDb
from utils import num_tokens_for_message
from contentclassifier import ContentClassifier

def get_urls(limit):
    """Get the URL, Title pairs from the history database
    """
    browser_profile_path  = os.getenv("EDGE_BROWSER_PROFILE_PATH")
    history_db = os.path.join(os.environ['HOME'], browser_profile_path, "History")
    
    print(f"Attempt loading history DB {history_db}")
    h = HistoryDb(history_db)
    urls = h.get_urls(limit=limit)
    return urls

def write_to_csv(csv_file, predictions, sep=","):
    """Write predictions to CSV
    """
    sep = ","
    header = ["\"Row\"","\"Title\"","\"Url\"", "\"Category\"","\"Confidence Level\""]
    with open(csv_file, "w") as f:
        f.write(sep.join(header))
        f.write("\n")    
        for idx, p in enumerate(predictions):
            if not p.endswith("\n"):
                p = "".join([p, "\n"])
            f.write(p)
       
def main(useAzure):
    """The main train
    """
  
    csv_file = "classifications.csv"
    urls = get_urls(limit=50) # decrease or increase limit to comfort level
    if urls:
        c = ContentClassifier(useAzure=useAzure)
        c.classify(urls)
        write_to_csv(csv_file, c.responses)       
    else:
        print(f"No URLs to classify. Check log for errors.")

if __name__ == "__main__":
    useAzure = True
    sys.exit(main(useAzure))