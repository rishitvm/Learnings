from autogen import AssistantAgent, register_function
from config import configure

def summary_generator():
    agent = AssistantAgent(
        name = "PaperSummarizerAgent",
        llm_config = {
            "config_list": configure(),
        },
        system_message = """You are a research assistant specialized in summarizing academic papers.
                            Given a list of papers with title, summary, and arXiv URL, generate concise 
                            and informative summaries for each paper. Do not fetch papers yourself. Do not 
                            call external APIs. Your task is to write clear, high-level summaries suitable 
                            for research overviews or literature reviews. Respond with 'TASK COMPLETE' after 
                            summarizing all papers."""
    )
    return agent