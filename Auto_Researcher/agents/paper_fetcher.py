from autogen import AssistantAgent, register_function
from tools.arxiv_search import fetch_papers
from config import configure

def create_paper_fetcher():
    agent = AssistantAgent(
        name = "PaperFetcherAgent",
        llm_config = {
            "config_list": configure(),
        },
        system_message = """You are a paper search expert. Use the registered tool to find recent arXiv papers. 
                            After providing the results, respond with 'TASK COMPLETE' message.
                         """
    )

    def get_arxiv_papers(topic: str):
        """Search arXiv and return recent papers related to a topic."""
        return fetch_papers(topic)
    
    register_function(
        f = get_arxiv_papers,
        caller = agent,
        executor = agent,
        name = "get_arxiv_papers",
        description = "Search arXiv and return recent papers related to a topic."
    )

    return agent