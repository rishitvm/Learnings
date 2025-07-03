import streamlit as st
import requests

st.set_page_config(page_title="Insurance Claim Clarifier", layout="centered")

st.title("Insurance Claim Clarification Assistant")

name = st.text_input("Your Name")
policy = st.text_input("Policy Number")
issue = st.text_area("Issue")

if st.button("Start Clarification Call"):
    with st.spinner("Contacting agent and clarifying..."):
        res = requests.post("http://127.0.0.1:8000/start-agent", json={
            "name": name,
            "policy": policy,
            "issue": issue
        })

        if res.status_code == 200:
            summary = res.json()["summary"]
            st.success("Clarification complete.")
            st.write("### Conversation Summary:")
            st.markdown(summary)
        else:
            st.error("Something went wrong.")
