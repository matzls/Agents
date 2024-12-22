from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
# api key from .env
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")



message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    ]
)
llm.invoke([message])