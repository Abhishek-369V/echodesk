"""
Minimal interactive demo of the EchoDesk agent — lets you type a customer
message and see the full pipeline output (intent, grounded reply, escalation
decision + reason) live, rather than only via CLI.
"""

import sys
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from src.agent.agent import EchoDeskAgent

st.set_page_config(page_title="EchoDesk", page_icon="🎧", layout="wide")

# Cache the agent instance so TF-IDF and embeddings don't reload on every button click
@st.cache_resource
def get_agent(use_llm_fallback: bool):
    return EchoDeskAgent(use_llm_escalation_fallback=use_llm_fallback)

st.title("🎧 EchoDesk - AppleSupport Triage Agent")
st.caption("Classify intent, draft a grounded reply, and decide escalation for an incoming customer message.")

use_hybrid = st.checkbox(
    "Use hybrid escalation (rule + LLM fallback)",
    value=False,
    help="Tested on the golden set: rule-only gets F1=0.58, hybrid gets F1=0.46(over-escalates). Off by default for that reason - see reports/failure_analysis.md.",
)

customer_text = st.text_area(
    "Customer message",
    placeholder="e.g. my bluetooth keeps disconnecting from my phone",
    height=100,
)

if st.button("Run agent", type="primary") and customer_text.strip():
    with st.spinner("Running classifier, retrieval, and escalation checks..."):
        agent = EchoDeskAgent(use_llm_escalation_fallback=use_hybrid)
        result = agent.handle(customer_text)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Intent", result["intent"])
        st.caption(f"Confidence: {result['intent_confidence']:.2f}")
    with col2:
        st.metric("Escalate?", "YES" if result["escalate"] else "NO")
        st.caption(f"Method: {result['escalation_method']}")

    st.subheader("Draft reply")
    st.info(result["draft_reply"])

    st.subheader("Escalation reasoning")
    for reason in result["escalation_reasons"]:
        st.write(f"- {reason}")

    st.subheader("Grounded on (retrieval)")
    st.caption(f"Top historical match similarity: {result['grounded_on_similarity']:.2f}")

st.divider()
st.caption(
    "Note: this is a demo built on a 176-example golden set - see reports/report.md for full evaluation, " \
    "known limitations, and failure analysis before treating any single output here as representative."
)