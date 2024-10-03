from langchain.chains import SimpleSequentialChain
from transformers import pipeline
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from bs4 import BeautifulSoup
import html2text
import re
from keybert import KeyBERT


#TODOs: improve email processing (classification, intent, summarizing etc); add agent to take actions (e.g. mark as read, trash, craft reply)

# Load environment variables from .env file
load_dotenv()

# Summarizing agent (uses Hugging Face model for summarization)
class SummarizingAgent:
    def __init__(self):
        # Initialize the summarization model from Hugging Face
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.kw_model = KeyBERT()
    
    def summarize(self, email_content):
        # Remove URLs
        email_content = re.sub(r'http\S+', '', email_content)
        email_content = re.sub(r'!\[.*?\]\(.*?\)', '', email_content)  # Remove image placeholders
        # email_content = re.sub(r'© 2019 Google LLC.*$', '', email_content, flags=re.S)  # Remove legal footer
        email_content = re.sub(r'\|.*\|', '', email_content)  # Remove tables/columns
        email_content = re.sub(r'---', '', email_content)  # Remove markdown-style separators
        email_content = ' '.join(email_content.split())

        if not email_content or len(email_content.split()) < 5:
            return "No content to summarize."  # Skip summarization for very short or empty emails
        try: 
            # clean_email_content = ' '.join(email_content.split())
            summary = self.summarizer(email_content, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except IndexError:
            print("Summarization failed, falling back to key phrase extraction.")
            return self.extract_key_phrases(email_content)
            # print(f"Problematic email content: {type(email_content)}")
            # clean_email_content = ' '.join(email_content.split())  # Remove extra whitespaces and newlines
            # print(f"Cleaned content: {clean_email_content}")
            # return "Failed to generate summary."
    def extract_key_phrases(self, email_content):
        # Use KeyBERT to automatically extract key phrases
        keyphrases = self.kw_model.extract_keywords(email_content, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        
        # Format the key phrases as a simple summary
        keyphrase_summary = ", ".join([phrase[0] for phrase in keyphrases])
        return f"Key highlights: {keyphrase_summary if keyphrases else 'No key phrases found.'}"
    
        # try:
        #     summary = self.summarizer(email_content, max_length=max_length, min_length=30, do_sample=False)
        #     return summary[0]['summary_text']
        # except IndexError:
        #     return "Failed to generate summary. The content may be too short or formatted incorrectly."
        # except Exception as e:
        #     return f"An error occurred: {str(e)}"


# Email pulling agent (uses Gmail API to fetch unread emails)
class EmailPullingAgent:
    def __init__(self):
        # Load credentials from the .env and client_secret.json
        self.creds = None
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        token_file = 'token.json'
        
        # Load credentials from file or generate new token
        if os.path.exists(token_file):
            try:
                self.creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            except ValueError as e:
                print(f"Error loading token.json: {e}")

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                # Start the OAuth flow using client_secret.json
                flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
                self.creds = flow.run_local_server(port=0)

            # Save the new credentials to token.json
            with open(token_file, 'w') as token:
                token.write(self.creds.to_json())        

        self.service = build('gmail', 'v1', credentials=self.creds)

    def extract_email_body(self, part):
        """Extract and clean the email body from various content types."""
        email_body = ''
        if 'parts' in part:
            for sub_part in part['parts']:
                email_body += self.extract_email_body(sub_part)  # Recursive call for each sub-part
        
        # If it's plaintext, decode and return it
        elif part['mimeType'] == 'text/plain' and 'data' in part['body']:
            email_body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')

        # If it's HTML, decode and convert it to text
        elif part['mimeType'] == 'text/html' and 'data' in part['body']:
            html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
            email_body += self.convert_html_to_text(html_content)

        return email_body
    
    def convert_html_to_text(self, html_content):
        """Convert HTML content to plain text."""
        soup = BeautifulSoup(html_content, 'html.parser')
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True  # Optional: ignore links in the conversion
        return text_maker.handle(str(soup))

    def fetch_unread_emails(self):
        # Fetch unread emails from Gmail
        results = self.service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])
        email_contents = {}

        for message in messages:
            msg = self.service.users().messages().get(userId='me', id=message['id']).execute()
            headers = msg['payload']['headers']
            sender = next(header['value'] for header in headers if header['name'] == 'From')
            subject = next(header['value'] for header in headers if header['name'] == 'Subject')
            # Extract the email body
            # parts = msg['payload'].get('parts', [])
            email_body = self.extract_email_body(msg['payload'])
            if not email_body:
                email_body = f"(Fallback to subject): {subject}"
            
            email_contents[sender] = email_body
            # if email_body:
            #     email_contents[sender] = email_body
            # else:
            #     email_contents[sender] = "No readable content found."

        return email_contents
    
    def fetch_todays_unread_emails(self):
        # Fetch today's unread emails only
        today_date = datetime.now().strftime("%Y/%m/%d")
        query = f"is:unread after:{today_date}"

        results = self.service.users().messages().list(userId='me', labelIds=['INBOX'], q=query).execute()
        messages = results.get('messages', []) #all emails
        email_contents = {}

        # for each eamil
        for message in messages:
            msg = self.service.users().messages().get(userId='me', id=message['id']).execute()
            headers = msg['payload']['headers']
            sender = next(header['value'] for header in headers if header['name'] == 'From')
            subject = next(header['value'] for header in headers if header['name'] == 'Subject')

            # Extract the email body, recursively handling nested parts
            email_body = self.extract_email_body(msg['payload'])

            # Fall back to subject if no readable body content is found
            if not email_body: #maybe this is where it's wrong? 
                print(f"{email_body} bad; falling back to subject: {subject}")
                email_body = f"(Fallback to subject): {subject}"
            
            email_contents[sender] = email_body

        return email_contents


# Orchestration agent (coordinates the email pulling and summarizing)
class OrchestrationAgent:
    def __init__(self, email_agent, summarizing_agent):
        self.email_agent = email_agent
        self.summarizing_agent = summarizing_agent

    def run(self):
        # Step 1: Fetch today's unread emails
        input("Press Enter to fetch today's unread emails...")
        emails = self.email_agent.fetch_todays_unread_emails()

        if not emails:
            print("No unread emails from today found.")
            return

        # Step 2: Summarize each email
        for sender, content in emails.items():
            input(f"Press Enter to summarize the email from {sender}...")
            summary = self.summarizing_agent.summarize(content)
            print(f"Summary of email from {sender}: {summary}\n")


# Instantiate the agents
email_agent = EmailPullingAgent()
summarizing_agent = SummarizingAgent()
orchestration_agent = OrchestrationAgent(email_agent, summarizing_agent)

# Run the orchestration agent to fetch and summarize emails
if __name__ == "__main__":
    orchestration_agent.run()
