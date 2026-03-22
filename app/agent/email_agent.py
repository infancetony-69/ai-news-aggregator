import os
from datetime import datetime
from typing import List, Optional
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class EmailIntroduction(BaseModel):
    greeting: str = Field(description="Personalized greeting with user's name and date")
    introduction: str = Field(description="2-3 sentence overview of the top ranked articles")


class RankedArticleDetail(BaseModel):
    digest_id: str
    rank: int
    relevance_score: float
    title: str
    summary: str
    url: str
    article_type: str
    reasoning: Optional[str] = None


class EmailDigestResponse(BaseModel):
    introduction: EmailIntroduction
    articles: List[RankedArticleDetail]
    total_ranked: int
    top_n: int

    def to_markdown(self) -> str:
        md = f"{self.introduction.greeting}\n\n{self.introduction.introduction}\n\n---\n\n"
        for article in self.articles:
            md += f"## {article.title}\n\n{article.summary}\n\n[Read more →]({article.url})\n\n---\n\n"
        return md


class EmailDigest(BaseModel):
    introduction: EmailIntroduction
    ranked_articles: List[dict] = Field(description="Top ranked articles with details")


EMAIL_SYSTEM = """You are an expert email writer for a daily AI news digest.

Write a warm, professional introduction that:
- Greets the user by name with the current date
- Gives a brief engaging overview of the top articles
- Is concise (2-3 sentences for introduction)

{format_instructions}"""


class EmailAgent:
    def __init__(self, user_profile: dict):
        self.user_profile = user_profile
        self.parser = PydanticOutputParser(pydantic_object=EmailIntroduction)
        self.llm = ChatCerebras(
            model="llama3.1-8b",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            temperature=0.7
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EMAIL_SYSTEM),
            ("human", "Create an email introduction for {name} on {date}.\n\nTop articles:\n{article_summaries}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def generate_introduction(self, ranked_articles: List) -> EmailIntroduction:
        current_date = datetime.now().strftime('%B %d, %Y')

        if not ranked_articles:
            return EmailIntroduction(
                greeting=f"Hey {self.user_profile['name']}, here is your daily digest for {current_date}.",
                introduction="No articles were available today."
            )

        top_articles = ranked_articles[:10]
        article_summaries = "\n".join([
            f"{i+1}. {a.title if hasattr(a, 'title') else a.get('title', 'N/A')} "
            f"(Score: {a.relevance_score if hasattr(a, 'relevance_score') else a.get('relevance_score', 0):.1f}/10)"
            for i, a in enumerate(top_articles)
        ])

        try:
            result = self.chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "name": self.user_profile["name"],
                "date": current_date,
                "article_summaries": article_summaries
            })
            return result
        except Exception as e:
            print(f"Error generating introduction: {e}")
            return EmailIntroduction(
                greeting=f"Hey {self.user_profile['name']}, here is your daily digest for {current_date}.",
                introduction="Here are the top AI news articles ranked by relevance to your interests."
            )

    def create_email_digest(self, ranked_articles: List[dict], limit: int = 10) -> EmailDigest:
        top_articles = ranked_articles[:limit]
        introduction = self.generate_introduction(top_articles)
        return EmailDigest(introduction=introduction, ranked_articles=top_articles)

    def create_email_digest_response(self, ranked_articles: List[RankedArticleDetail], total_ranked: int, limit: int = 10) -> EmailDigestResponse:
        top_articles = ranked_articles[:limit]
        introduction = self.generate_introduction(top_articles)
        return EmailDigestResponse(
            introduction=introduction,
            articles=top_articles,
            total_ranked=total_ranked,
            top_n=limit
        )
