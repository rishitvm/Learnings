from autogen import UserProxyAgent
from config import configure

def create_planner():
    return UserProxyAgent(
        name = "PlannerAgent",
        human_input_mode = "NEVER",
        max_consecutive_auto_reply = 1,
        is_termination_msg = lambda x: "TASK COMPLETE" in x.get("content", ""),
        system_message = """You are the planner. Break down the user's request and delegate to the correct agent. 
                            Respond with 'TASK COMPLETE' to end the conversation.""",
        llm_config = {"config_list": configure()},
        code_execution_config = {"use_docker": False},
    )



