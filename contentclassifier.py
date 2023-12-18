import os
from utils import num_tokens_for_message

class ContentClassifier:
    """
    """
    def __init__(self, useAzure=False
                ,model="gpt-3.5-turbo-0613", deployment_name="gpt35t-0613") -> None:
        
        self.useAzure = useAzure
        self.model = model
        self.deployment_name = deployment_name
        #get the right client if Azure true/false
        self.client = self.create_api_client(self.useAzure)
        #self.contents = contents
        self.system_content = open("./system-prompt-csv.txt").read()
        self.responses = []
        self.row_count = 0
    
    @staticmethod
    def create_api_client(useAzure):
        if useAzure:
            from openai import AzureOpenAI
            return AzureOpenAI()
        else:
            from openai import OpenAI
            return OpenAI()

    def classify(self, url_data):
        """Generate messages to send, keeping track of max_tokens, and ask for predictions.
        """
        messages = []
        max_tokens = 2000
        num_tokens = 0
        print(f"Rows found {len(url_data)}")
        sys_message = {"role": "system", "content": self.system_content}
        sys_tokens = num_tokens_for_message(sys_message, self.model)
        num_tokens += sys_tokens
        messages.append(sys_message)

        for index, row in enumerate(url_data):
            url, title = row
            self.row_count+=1
             
            content = f"{self.row_count}, URL: {url}, TITLE: {title}"
            message = {"role" : "user", "content" : content}
            num_tokens += num_tokens_for_message(message, self.model)
            if num_tokens <= max_tokens:
                messages.append(message)
            else:
                # We've reached already reached our token limit
                print(f"Token limit reached at {index}...")
                # let's first get an intermediate response before moving on with this message
                msg_count = len(messages)
                print(f"Sending {msg_count} messages for classification...")
                # grab the contents of the completion response and append it to the responses list
                r = self.get_completion(messages) 
                self.responses.append(r)
                num_tokens = num_tokens_for_message(message, self.model)
                # message queue reset for the the next batch of messages, 
                # reset sys message and append last message
                messages = []
                messages.append(sys_message)
                num_tokens += sys_tokens
                messages.append(message)
        
        # no more messages to append
        all_sent_message = ({"role": "user", "content" : "ALL SENT"})
        num_tokens += num_tokens_for_message(all_sent_message, self.model)
        messages.append(all_sent_message)
        # get the last response (which is also the first if token limit was never reached)
        msg_count = len(messages)
        print(f"Sending {msg_count} messages for classification...")
        final_r = self.get_completion(messages)
        # add the content                
        self.responses.append(final_r)
        #print(f"Newlines in responses {self.responses.count('\n')}")   
    
    def get_completion(self, messages, temperature=0.3):
        """
        """
        temperature = temperature
        # Azure needs deployment name as model name
        model = (self.model, self.deployment_name)[self.useAzure]
        response = self.client.chat.completions.create(model = model
                                                      ,temperature=temperature
                                                      ,messages=messages)
        r = response.choices[0].message.content
        return r

if __name__ == "__main__":
    content = [("URL: https://www.msn.com", "TITLE: MSN.com"), ("URL: https://www.bing.com", "TITLE: Bing Search")]
    c = ContentClassifier(useAzure=True)
    print(c)
    c.classify(content)
    for p in c.responses:
        print(p)