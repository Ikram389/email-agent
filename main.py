import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    HumanInTheLoopMiddleware,
    dynamic_prompt,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import tool, ToolRuntime
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command


 
# 1. ENVIRONMENT
 

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from .env")


 
# 2. STATE
 

class EmailAgentState(AgentState):
    """
    State belonging to the current conversation.

    authenticated:
        False -> user has not authenticated
        True  -> user has authenticated
    """

    authenticated: bool


 
# 3. CONTEXT
 

class EmailContext:
    """
    Context contains information about the user/environment.

    This is NOT conversation state.

    These are the credentials against which we authenticate.
    """

    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password


 
# 4. AUTHENTICATION TOOL
 

@tool
def authenticate(
    email: str,
    password: str,
    runtime: ToolRuntime[EmailContext, EmailAgentState],
) -> Command:
    """
    Authenticate the user.

    The supplied email and password are compared against
    the credentials stored in the runtime context.
    """

    expected_email = runtime.context.email
    expected_password = runtime.context.password

    if (
        email == expected_email
        and password == expected_password
    ):
        return Command(
            update={
                "authenticated": True
            },
            messages=[
                {
                    "role": "tool",
                    "content": "Authentication successful."
                }
            ],
        )

    return Command(
        update={
            "authenticated": False
        },
        messages=[
            {
                "role": "tool",
                "content": "Authentication failed. Invalid email or password."
            }
        ],
    )


 
# 5. READ EMAIL TOOL
 

@tool
def read_email() -> str:
    """
    Read the user's latest email.

    This is a dummy mailbox. No real email service is contacted.
    """

    return """
    EMAIL

    From: boss@example.com
    To: user@example.com
    Subject: Project update

    Hi,

    Could you send me the latest project status?

    Thanks,
    Mohamed
    """


 
# 6. SEND EMAIL TOOL
 

@tool
def send_email(
    to: str,
    subject: str,
    body: str,
) -> str:
    """
    Send an email.

    This is a dummy email sender.
    It does NOT send a real email.

    Human approval is required before this tool can execute.
    """

    print("\n" + "=" * 60)
    print("DUMMY EMAIL SENT")
    print("=" * 60)
    print(f"To:      {to}")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print("=" * 60)

    return "Email successfully sent."


 
# 7. DYNAMIC TOOL MIDDLEWARE
 

@wrap_model_call
def dynamic_tools(
    request: ModelRequest,
    handler,
) -> ModelResponse:

    authenticated = request.state.get(
        "authenticated",
        False
    )

    if authenticated:

        # User is authenticated.
        # Give the model all three tools.

        request = request.override(
            tools=[
                authenticate,
                read_email,
                send_email,
            ]
        )

    else:

        # User is NOT authenticated.
        # The model receives ONLY authenticate.

        request = request.override(
            tools=[
                authenticate
            ]
        )

    return handler(request)


 
# 8. DYNAMIC PROMPT
 

@dynamic_prompt
def dynamic_system_prompt(
    request: ModelRequest,
) -> str:

    authenticated = request.state.get(
        "authenticated",
        False
    )

    if not authenticated:

        return """
You are a secure email gateway.

The user has NOT authenticated yet.

Your ONLY responsibility is to help the user authenticate.

You must NOT:
- read emails
- send emails
- reveal email contents
- pretend the user is authenticated
- bypass authentication
- use tools other than authenticate

If the user asks to read or send an email before
authentication, politely explain that authentication
is required first.

Ask the user for their email and password when necessary.
"""

    return """
You are a helpful email assistant.

The user has successfully authenticated.

You can:
- read emails
- help understand emails
- prepare replies
- send replies

IMPORTANT:

Sending an email is a consequential action.

A human approval step will happen before an email
is actually sent. Never claim that an email was sent
unless the send_email tool confirms it.

Do not bypass the human approval process.
"""


 
# 9. MODEL
 

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


 
# 10. CREATE AGENT
 

checkpointer = InMemorySaver()


agent = create_agent(
    model=model,

    tools=[
        authenticate,
        read_email,
        send_email,
    ],

    middleware=[
        dynamic_tools,
        dynamic_system_prompt,

        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True
            }
        ),
    ],

    state_schema=EmailAgentState,

    context_schema=EmailContext,

    checkpointer=checkpointer,
)


 
# 11. USER CONTEXT
 

context = EmailContext(
    email="ikram@example.com",
    password="secret123",
)


 
# 12. CONVERSATION CONFIGURATION
 

config = {
    "configurable": {
        "thread_id": "email-demo-1"
    }
}


 
# 13. HELPER FUNCTION
 

def print_last_message(result):

    messages = result.get("messages", [])

    if not messages:
        return

    last_message = messages[-1]

    print("\nAGENT:")
    print(last_message.content)



# 14. MAIN LOOP


def main():

    print("=" * 60)
    print("SECURE EMAIL AGENT")
    print("=" * 60)

    print("""
This is a dummy email system.

Expected credentials:

Email:    ikram@example.com
Password: secret123

Nothing will touch a real mailbox.
""")

    while True:

        user_input = input("\nYOU: ")

        if user_input.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Goodbye!")
            break

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
                "authenticated": False,
            },
            config=config,
            context=context,
        )

         
        # HUMAN-IN-THE-LOOP
         

        if "__interrupt__" in result:

            print("\n" + "=" * 60)
            print("HUMAN APPROVAL REQUIRED")
            print("=" * 60)

            interrupts = result["__interrupt__"]

            for interrupt in interrupts:

                print(
                    "\nThe agent wants to perform "
                    "the following action:"
                )

                print(interrupt.value)

            answer = input(
                "\nApprove this action? (yes/no): "
            ).strip().lower()

            if answer in {
                "yes",
                "y",
            }:

                print("\nApproved.")

                result = agent.invoke(
                    Command(
                        resume={
                            "decisions": [
                                {
                                    "type": "approve"
                                }
                            ]
                        }
                    ),
                    config=config,
                    context=context,
                )

            else:

                print("\nRejected.")

                result = agent.invoke(
                    Command(
                        resume={
                            "decisions": [
                                {
                                    "type": "reject",
                                    "message": (
                                        "The user rejected "
                                        "sending the email."
                                    ),
                                }
                            ]
                        }
                    ),
                    config=config,
                    context=context,
                )

        print_last_message(result)


 
# 15. RUN
 

if __name__ == "__main__":
    main()