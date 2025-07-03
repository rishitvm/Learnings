import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model = "llama-3.1-8b-instant", api_key = GROQ_API_KEY)

def build_prompt(doc_id, name, policy, history):
    prompt = f"""
                You are an AI assistant from the insurance claim department. You are speaking with an insurance agent to clarify the current status of a patient's insurance claim.

                Patient Information:
                - Patient ID: {doc_id}
                - Name: {name}
                - Policy ID: {policy}

                Your goals:
                1. Start with a polite greeting and ask for the **current claim status**.
                2. If th statis is **Accepted** need not ask more question may be just one follow up question like were all
                   the documents were good like that and after answering you can end up. Don't prolong the conversation.
                3. If the status is **Rejected**, **On Hold**, or **Needs Changes**, ask for the **specific reason**.
                4. If the reason is vague or unclear, ask follow-up questions for clarity. However, if the agent clearly 
                   says “yes” or “everything is good,” treat that as a full confirmation unless otherwise ambiguous.
                5. Once you have **complete clarity**, say a **thank-you or polite closing statement** and end the conversation.

                Rules:
                - Do **not repeat** the patient's info unless asked.
                - If the agent says "rejected", "approved", or "on hold", ask meaningful follow-ups if needed.
                - Use natural, professional tone.
                - End only when you’re satisfied with the response (e.g., you have both status and a clear reason).
                - Always end with a polite closing like “Thank you for your help. Have a great day.”

                Once you are fully satisfied and ready to end the conversation, respond with your final thank-you or 
                closing message, and make sure to end that sentence with: $end

                "$end" should only be added at the very end of the conversation, after you've asked all 
                necessary follow-up questions and received complete clarity. It must be the last token in 
                your final message. Do NOT include $end in any questions or in the middle of the conversation.

                Conversation so far:
             """
    prompt += history + "ai:"
    return prompt

def call_llm(prompt: str):
    llm_response = llm.stream(prompt)
    for chunk in llm_response:
        if chunk.content:
            yield chunk.content
