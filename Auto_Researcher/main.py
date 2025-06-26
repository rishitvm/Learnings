from agents.planner import create_planner
from agents.paper_fetcher import create_paper_fetcher
from agents.summarizer import summary_generator
from autogen import GroupChat, GroupChatManager

planner = create_planner()
paper_fetcher = create_paper_fetcher()
paper_summarizer = summary_generator()

group_chat = GroupChat(
    agents = [planner, paper_fetcher, paper_summarizer],
    messages = [],
    max_round = 10,
    speaker_selection_method = "round_robin",
    allow_repeat_speaker = False
)

manager = GroupChatManager(
    groupchat = group_chat,
    llm_config = {"config_list": planner.llm_config["config_list"]}
)

planner.initiate_chat(
    manager,
    message = """Find 3 recent papers about 'multimodal AI'. Then, summarize each one briefly with main contributions, suitable for a literature review. 
Do not re-fetch any paper once found. Just use the paper title, summary, and url from the search result. 
After summarizing, say TASK COMPLETE."""
)
