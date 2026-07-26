
            ## Basic Plan for the assignment ##                         

### tool --> sends email --> subject, body, to
### this credentials.json is having our consent to use email address that we setup
### from google cloud console
### so we need a method to use credentials.json and authorize the service
# System prompt
# agent --> tool wireup
# done

# importing the required libraries
import os
import base64
from email.message import EmailMessage
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail scope: this sends only. If later want to read/draft, add more scopes and delete token.json so the consent flow re-runs and run again.
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# defined environment varibale file .env where the grow api token is stored
load_dotenv()
# Paths to the OAuth client secrets file (from Google Cloud Console) and the cached token file
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"


def _get_gmail_service():
    """Load cached OAuth credentials, refreshing or running the consent flow as needed."""
    creds = None

    # trying to load previously saved credentials from disk
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # if no creds, or they're invalid/expired,then this handle accordingly
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_FILE}. Download an OAuth 2.0 'Desktop app' "
                    "client from the Google Cloud Console and save it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            # Opens a browser once for consent; token is cached to token.json afterward.
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email from the user's Gmail account.

    Args:
        to: Recipient email address.
        subject: Subject line of the email.
        body: Plain-text body content of the email.

    Returns:
        A dict with a 'status' key ('success' or 'error') and a 'detail' message.
        On success it also includes the Gmail 'message_id'.
    """
    try:
        # Gets an authorized Gmail service
        service = _get_gmail_service()
        # Builds the email message using Python's built-in EmailMessage class
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to
        message["Subject"] = subject

        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
        # Calls the Gmail API to actually send the message
        sent = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": encoded})
            .execute()
        )
        # Returns a success result including Gmail's message ID
        return {
            "status": "success",
            "message_id": sent.get("id"),
            "detail": f"Email sent to {to}.",
        }
    except Exception as e:  # Catchs any auth error, network error, or API error lands here
        return {"status": "error", "detail": str(e)}

# the root agnet that uses the sned email tool also an instruction prompt describing how the agent should behave.
root_agent = Agent(
    name="email_agent",
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    description="An agent that composes and sends emails on the user's behalf via Gmail.",
    instruction=(
        "You are a helpful email assistant whose name is ROO. When the user asks to send an email, "
        "gather the recipient address, subject, and body. If any of those are "
        "missing, ask concise follow-up questions.\n\n"
        "Before sending, ALWAYS show the user a draft (To, Subject, Body) and ask "
        "for explicit confirmation. Only call the `send_email` tool after the user "
        "clearly confirms. After sending, report the result — success with the "
        "message id, or the error detail if it failed."
    ),
    tools=[send_email],
)
