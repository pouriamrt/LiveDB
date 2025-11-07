from Config import config
from loguru import logger
from agno.models.openai import OpenAIChat
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from agno.db.postgres import PostgresDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.agent import Agent
from agno.tools.reasoning import ReasoningTools
from agno.tools.postgres import PostgresTools
from urllib.parse import urlparse, unquote

logger.info("Agents initialized")

model = OpenAIChat(id=config.MODEL_NAME, api_key=config.OPENAI_API_KEY)

contents_db = PostgresDb(
    db_url=config.PGVECTOR_CONTENTS_URL,
    knowledge_table=config.PGVECTOR_CONTENTS_TABLE
)

vector_db=PgVector(table_name=config.PGVECTOR_TABLE, db_url=config.PGVECTOR_URL, search_type=SearchType.hybrid, 
                       embedder=OpenAIEmbedder(id=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY))


knowledge_base = Knowledge(
    vector_db=vector_db,
    contents_db=contents_db
)

memory_db = PostgresDb(db_url=config.PGVECTOR_MEMORY_URL, memory_table=config.PGVECTOR_MEMORY_TABLE)

# Reasoning Agent
reasoning_agent = Agent(
    name="ReasoningAgent",
    description="You are a logical reasoning assistant that breaks down complex problems",
    model=model,
    db=memory_db,
    instructions=[
        "You are a logical reasoning assistant that breaks down complex problems",
        "Use step-by-step thinking to analyze situations thoroughly",
        "Apply structured reasoning to reach well-founded conclusions",
        "Show your reasoning process clearly to help users understand your logic",
    ],
    tools=[ReasoningTools()],
    markdown=True,
    exponential_backoff=True,
    # debug_mode=True,
    read_chat_history=True,
    max_tool_calls_from_history=3,
    add_history_to_context=True,
    num_history_runs=3,
)


# Knowledge Agent
knowledge_agent = Agent(
    name="KnowledgeAgent",
    model=model,
    knowledge=knowledge_base,
    db=memory_db,
    description="You are a knowledge assistant that uses the knowledge base to answer questions.",
    instructions=[
        "Answer questions comprehensively based on the knowledge base.",
        "Summarize key points and include the source URL.",
        "If the question is not in the knowledge base, say so and ask the user to provide more information.",
        "If the question is not clear, ask the user to provide more information.",
        "Make the response nice and beautifully formatted as well with sections and inline citations."
    ],
    expected_output="Answer the question comprehensively based on the context provided with citations and references. If the information is not available, say 'I don't know'.",
    max_tool_calls_from_history=3,
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    markdown=True,
    add_knowledge_to_context=True,
    search_knowledge=True,
    enable_agentic_knowledge_filters=True,
    exponential_backoff=True,
    # debug_mode=True,
)

# SQL DB
normalized = config.SQL_DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

u = urlparse(normalized)
postgres_tools = PostgresTools(
    host=u.hostname,
    port=u.port or 5432,
    db_name=u.path.lstrip("/"),
    user=unquote(u.username or ""),
    password=unquote(u.password or "")
)

# SQL Agent
sql_agent = Agent(
    name="SQLAgent",
    model=model,
    db=memory_db,
    tools=[postgres_tools],
    description="You are a SQL agent that uses the postgres database to answer questions.",
    instructions=[
        "Use the postgres database to answer questions.", 
        "Always check the database schema and first 2 rows of each table to decide which tables to use before running the query.",
        "If you encounter an error, try to fix it by checking the database schema and the first 2 rows of each table.",
        "If the question is not clear, ask the user to provide more information."
    ],
    expected_output="Answer the question comprehensively. If the information is not available, say 'I don't know'.",
    max_tool_calls_from_history=3,
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    markdown=True,
    exponential_backoff=True,
    # debug_mode=True,
)

# General Agent
general_agent = Agent(
    name="GeneralAgent",
    model=model,
    db=memory_db,
    add_history_to_context=True,
    num_history_runs=4,
    description="You are a general agent that handles general queries and synthesizes info from specialists.",
    instructions=[
        "Answer general questions or combine inputs from specialist agents.",
        "If specialists provide info, synthesize it into a clear answer.",
        "If a query doesn't fit other specialists, attempt to answer directly.",
        "Maintain a professional and clear tone.",
        "Make the response beautifully formatted as well with sections and inline citations."
    ],
    max_tool_calls_from_history=3,
    markdown=True,
    exponential_backoff=True,
    # debug_mode=True
)

