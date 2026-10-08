from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx
import streamlit as st

from ui_contract import API_PATHS, load_personas, select_persona


API_URL = os.getenv("API_URL", "http://localhost:8000")


def request(method: str, path: str, token: str, **kwargs):
    headers = {"Authorization": f"Bearer {token}"}
    response = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=40, **kwargs)
    if response.is_error:
        detail = response.json().get("detail", response.text)
        st.error(f"API {response.status_code}: {detail}")
        return None
    return response.json()


st.set_page_config(page_title="Asisten HR Nusantara Digital", page_icon="🧭", layout="wide")
st.title("Asisten Kebijakan HR dan Cuti")
st.caption("Data dan token pada aplikasi ini sepenuhnya sintetis untuk kebutuhan pembelajaran.")

personas = load_personas(Path(__file__).with_name("demo_personas.json"))
persona = st.sidebar.selectbox("Persona demo", personas, format_func=lambda item: item["label"])
select_persona(st.session_state, persona)
st.sidebar.caption(persona["scenario"])
token = persona["demo_token"]

chat_tab, leave_tab, monitor_tab = st.tabs(["Percakapan", "Permintaan Cuti", "Monitoring"])

with chat_tab:
    message = st.text_area("Pertanyaan atau permintaan", placeholder="Berapa saldo cuti saya?")
    if st.button("Kirim", type="primary", disabled=not message.strip()):
        result = request("POST", API_PATHS["chat"], token, json={"message": message})
        if result:
            st.session_state["last_chat"] = result
    result = st.session_state.get("last_chat")
    if result:
        st.markdown(result["answer"])
        st.caption(f"Outcome: {result['outcome']} · Trace: {result['trace_id']}")
        if result["citations"]:
            with st.expander("Sumber kebijakan"):
                for citation in result["citations"]:
                    st.markdown(f"**{citation['document_id']} — {citation['section']}**")
                    st.write(citation["excerpt"])
        proposal = result.get("action_proposal")
        if proposal:
            st.warning("Tindakan belum dijalankan. Periksa detail lalu konfirmasikan.")
            st.json(proposal)
            key_name = f"idem_{proposal['proposal_id']}"
            st.session_state.setdefault(key_name, str(uuid.uuid4()))
            if st.button("Konfirmasi pembuatan permintaan pending", key=proposal["proposal_id"]):
                confirmed = request(
                    "POST",
                    API_PATHS["confirm"].format(proposal_id=proposal["proposal_id"]),
                    token,
                    json={"idempotency_key": st.session_state[key_name]},
                )
                if confirmed:
                    st.success(f"Permintaan {confirmed['request_id']} berstatus {confirmed['status']}.")

with leave_tab:
    if st.button("Muat permintaan saya"):
        data = request("GET", API_PATHS["my_requests"], token)
        if data is not None:
            st.dataframe(data, use_container_width=True)

with monitor_tab:
    st.info("Gunakan token monitoring admin untuk membuka metrik dan alert agregat.")
    if st.button("Muat monitoring"):
        summary = request("GET", API_PATHS["monitoring_summary"], token)
        alerts = request("GET", API_PATHS["monitoring_alerts"], token)
        if summary:
            columns = st.columns(4)
            columns[0].metric("Runs", summary["run_count"])
            columns[1].metric("Error rate", f"{float(summary['error_rate']):.1%}")
            columns[2].metric("p95 latency", f"{int(summary['p95_latency_ms'])} ms")
            columns[3].metric("Tool calls", summary["tool_calls"])
        if alerts is not None:
            st.subheader("Alert aktif")
            st.json(alerts)

