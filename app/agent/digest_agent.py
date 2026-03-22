import os
from typing import Optional
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class DigestOutput(BaseModel):
    title: str
    summary: str


DIGEST_SYSTEM = """You are an expert AI news analyst. Create concise digests for AI-related content.

Guidelines:
- Title: 5-10 words capturing the essence
- Summary: 2-3 sentences highlighting main points and why they matter
- Focus on actionable insights, avoid marketing fluff

{format_instructions}"""


class DigestAgent:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=DigestOutput)
        self.llm = ChatCerebras(
            model="llama3.1-8b",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            temperature=0.7
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", DIGEST_SYSTEM),
            ("human", "Create a digest for this {article_type}:\nTitle: {title}\nContent: {content}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def generate_digest(self, title: str, content: str, article_type: str) -> Optional[DigestOutput]:
        try:
            return self.chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "article_type": article_type,
                "title": title,
                "content": content[:8000]
            })
        except Exception as e:
            print(f"Error generating digest: {e}")
            return None
