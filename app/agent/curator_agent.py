import os
from typing import List
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class RankedArticle(BaseModel):
    digest_id: str = Field(description="The ID of the digest (article_type:article_id)")
    relevance_score: float = Field(description="Relevance score from 0.0 to 10.0", ge=0.0, le=10.0)
    rank: int = Field(description="Rank position (1 = most relevant)", ge=1)
    reasoning: str = Field(description="Brief explanation of why this article is ranked here")


class RankedDigestList(BaseModel):
    articles: List[RankedArticle] = Field(description="List of ranked articles")


CURATOR_SYSTEM = """You are an expert AI news curator specializing in personalized content ranking for AI professionals.

Rank articles from most relevant (rank 1) to least relevant based on the user profile below.

Scoring Guidelines:
- 9.0-10.0: Highly relevant, directly aligns with user interests
- 7.0-8.9: Very relevant, strong alignment
- 5.0-6.9: Moderately relevant
- 3.0-4.9: Somewhat relevant
- 0.0-2.9: Low relevance

User Profile:
Name: {name}
Background: {background}
Expertise Level: {expertise_level}
Interests: {interests}
Preferences: {preferences}

{format_instructions}"""


class CuratorAgent:
    def __init__(self, user_profile: dict):
        self.user_profile = user_profile
        self.parser = PydanticOutputParser(pydantic_object=RankedDigestList)
        self.llm = ChatCerebras(
            model="llama3.1-8b",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            temperature=0.3
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CURATOR_SYSTEM),
            ("human", "Rank these {count} AI news digests:\n\n{digest_list}\n\nProvide a relevance score and rank for each article.")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def rank_digests(self, digests: List[dict]) -> List[RankedArticle]:
        if not digests:
            return []

        digest_list = "\n\n".join([
            f"ID: {d['id']}\nTitle: {d['title']}\nSummary: {d['summary']}\nType: {d['article_type']}"
            for d in digests
        ])

        interests = ", ".join(self.user_profile.get("interests", []))
        preferences = str(self.user_profile.get("preferences", {}))

        try:
            result = self.chain.invoke({
                "name": self.user_profile["name"],
                "background": self.user_profile["background"],
                "expertise_level": self.user_profile["expertise_level"],
                "interests": interests,
                "preferences": preferences,
                "format_instructions": self.parser.get_format_instructions(),
                "count": len(digests),
                "digest_list": digest_list
            })
            return result.articles if result else []
        except Exception as e:
            print(f"Error ranking digests: {e}")
            return []
