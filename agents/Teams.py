from Config import config
from loguru import logger
from agents.Agents import reasoning_agent, knowledge_agent, sql_agent, general_agent, memory_db, model
from agno.team import Team

def initialize_team(session_state: dict) -> Team:
    """Initializes or re-initializes the research assistant team."""
    logger.info("Teams initialized")
    
    return Team(
        name="ResearchAssistantTeam",
        model=model, 
        members=[
            knowledge_agent,
            sql_agent,
            reasoning_agent,
            general_agent,
        ],
        description="You are a coordinator that coordinates a team of specialist agents to handle research tasks.",
        instructions=[
            "Analyze the user query and decide which specialist(s) should handle it.",
            "Delegate tasks based on query type:",
            "- Knowledge -> KnowledgeAgent",
            "- SQL queries -> SQLAgent",
            "- Reasoning -> ReasoningAgent",
            "- General or multi-step queries -> GeneralAgent",
            "Gather all agents' findings and synthesize a coherent answer.",
            "Cite sources for any facts and maintain clarity in the final answer.",
            "In the final response, include a clickable hyperlink to the reference.",
            "Always check the conversation history (memory) for context or follow-up references.",
            "If the user asks something that was asked before, utilize remembered information instead of starting fresh.",
            "Continue delegating and researching until the query is fully answered.",
            "Only if the user asks to reason, use ReasoningAgent based on the answer of the other agents.",
            "Avoid mentioning the function calls in the final response and make the final response beautifully formatted as well."
        ],
        db=memory_db,
        session_state=session_state,
        expected_output="The user's query has been thoroughly answered with information from all relevant agents.",
        enable_agentic_state=True,      # The coordinator retains its own context between turns
        share_member_interactions=True, # All agents see each other's outputs as context
        enable_agentic_memory=True,
        enable_user_memories=True,
        read_team_history=True,
        show_members_responses=False,   # Do not show raw individual agents' answers directly to the user
        delegate_task_to_all_members=False,
        markdown=True,
        add_member_tools_to_context=True,
        add_memories_to_context=True,
        add_history_to_context=True,    # Maintain a shared history (memory) between coordinator and members
        num_history_runs=4,             # Limit how much history is shared (to last 3 interactions)
        add_session_state_to_context=True,
        exponential_backoff=True,
        # debug_mode=True
    )