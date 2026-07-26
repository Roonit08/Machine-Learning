# Email Agent (ROO)


## Description of the Tool

It is a small agent of artificial intelligence that helps me to send emails through my Gmail account. I named the agent ROO. All it takes is me telling the agent whom to send an email, the subject of the email, and the content, and it would write and send it automatically through Gmail.


## Reason for Building it

First of all, I wanted to learn more about the implementation of an artificial intelligence agent into a real-life tool, so that it could do some actions apart from just processing. And sending an email was a nice choice because it is familiar to everybody and involves authentication process rather than a simple function call.


## How it works, in plain terms

The Agent Development Kit (ADK) by Google was utilized for the development of the agent. It is a framework which helps to develop the personality of the agent and enables it to work with some tools.

Regarding the "brain" of the agent, that is the language model which understands the user's request and takes actions accordingly, I integrated it with the Groq API. I used the LLaMA 3.3 70B model from Groq, thus the cognitive capabilities of the agent are provided by Groq while the action side belongs to Google's framework.

In order to actually send emails, I created a project in Google Cloud Console and enabled the Gmail API there. Google Cloud gives the user a credentials file, which is a permission document which proves that the application can ask for access to the Gmail account. The first time when the agent sends an email, it will open a web browser and asks the user to authenticate the application's access to his email. The permission is stored in the token file and no authentication is required after that anymore.

The tool at its disposal is only one, which is sending an email. Whenever I request it to send anything, it doesn’t send it immediately. Instead, it presents me with a draft, which includes the recipient, the subject, and the body of the email. Once I have confirmed it, it will then proceed to send it via Gmail. After sending it, it informs me whether it was successful and even provides me with a confirmation ID in case of success or provides an error message in case of failure.


## Conclusion

I have built an AI-assistant which can engage in a short dialogue with me in order to get all the necessary information about the email to be sent, display the prepared draft, wait for the confirmation of the correctness of the email and finally send the email to somebody's inbox via my own Gmail account. This is quite a tiny project, but it allows seeing the whole process from the generation of an AI decision to the actual real-world action performed.



## What is inside the project folder

- The agent itself: both the email-sending utility and the agent's instructions and behavior.

- The credentials file obtained via the Google Cloud Console, which is necessary for the application in order to ask for Gmail access.

- Token file which stores the login data after the very first time of using the application and doesn't require relog in afterwards.

- Environment file with the Groq API key stored outside the main code.

- The requirements file with the list of necessary libraries.



## Possible Future Directions

For the moment the only functionality that the agent has is email sending capabilities. In the future I would expand on this by allowing it to read received emails, respond to them automatically, and even manage an inbox.
