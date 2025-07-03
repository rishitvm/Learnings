from play import enqueue_audio, start_audio_thread, stop_audio_thread, audio_queue
from tts import call_tts
from llm import call_llm, build_prompt
from mongo import get_all_patients
from asr import call_asr
import time
from pymongo import MongoClient
from bson import ObjectId
from RealtimeSTT.audio_recorder import AudioToTextRecorder
from TTS.api import TTS

client = MongoClient("mongodb://localhost:27017/")
collection = client["ai"]["insurance_data"]

recorder = AudioToTextRecorder(model='tiny', language='en')
tts_model = TTS(model_name="tts_models/en/vctk/vits")

def run(doc):
    start_audio_thread()
    print(f"\nStarting conversation for: {doc['name']} ({doc['policy']})")
    doc_id = doc["_id"]
    history = doc.get("history", "")

    chunk_count = 0
    status = None
    reason = ""

    if history == "":
        history = "ai:"

    while True:
        prompt = build_prompt(doc_id, doc["name"], doc["policy"], history)
        buffer = ""
        punctuation_marks = [".", "!", "?"]

        for token in call_llm(prompt):
            print(token, end="", flush=True)
            buffer += token
            time.sleep(0.01)

            if token.strip() in punctuation_marks:
                chunk = buffer.strip()
                if chunk:
                    history += chunk
                    file = call_tts(tts_model, chunk, chunk_count)
                    enqueue_audio(file)
                    chunk_count += 1
                    buffer = ""

        if buffer.strip():
            chunk = buffer.strip()
            history += chunk
            file = call_tts(tts_model, chunk, chunk_count)
            enqueue_audio(file)
            chunk_count += 1

        audio_queue.join()

        if "$end" in history.lower():
            print("\nLLM indicated end of conversation.")
            break

        print("\n------------------------------------------------------")
        agent_response = call_asr(recorder)
        print("You:", agent_response)
        print("------------------------------------------------------")

        history += "\nagent: " + agent_response
        lower_resp = agent_response.lower()

        if status is None:
            if "rejected" in lower_resp:
                status = "Rejected"
                continue
            elif "accepted" in lower_resp or "approved" in lower_resp:
                status = "Accepted"
                reason = "Approved successfully"
                continue
            elif "on hold" in lower_resp or "still pending" in lower_resp:
                status = "On Hold"
                continue
            elif "change" in lower_resp or "needs changes" in lower_resp:
                status = "Needs Changes"
                continue

        elif reason == "":
            reason = agent_response
            continue

    collection.update_one({"_id": doc_id}, {
        "$set": {
            "status": status,
            "reason": reason,
            "history": history
        }
    })

    summary_prompt = f"""Here is the user info:\nName: {doc['name']}\nPolicy: {doc['policy']}\nConversation: {history}\nSummarize this."""
    summary_output = ""
    print("------------ Summary -------------")
    for token in call_llm(summary_prompt):
        print(token, end="", flush=True)
        summary_output += token

    collection.update_one({"_id": doc_id}, {"$set": {"summary": summary_output}})

    stop_audio_thread()
    print("\nConversation Completed")
