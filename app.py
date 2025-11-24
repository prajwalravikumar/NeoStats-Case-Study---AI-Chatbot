import streamlit as st
import os
import glob
import pandas as pd
import logging
from utils.web_search import serpapi_search
from models.llm import ask_llm
from utils.pdf_reader import parse_pdf_to_chunks

from rapidfuzz import process, fuzz

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="KARNA - The KSRTC Assistant", layout="centered")

SYSTEM_PROMPT = "You are a helpful KSRTC schedule assistant."

UPLOADED_KBS = "/mnt/data/KSRTC Bus time table from KBS.pdf"
UPLOADED_SAT = "/mnt/data/KSRTC Bus time table from Satellite.pdf"

def find_pdf_paths():
    local_pdfs = glob.glob(os.path.join("pdfs", "*.pdf"))
    if os.path.exists(UPLOADED_KBS) and UPLOADED_KBS not in local_pdfs:
        local_pdfs.append(UPLOADED_KBS)
    if os.path.exists(UPLOADED_SAT) and UPLOADED_SAT not in local_pdfs:
        local_pdfs.append(UPLOADED_SAT)
    return local_pdfs

@st.cache_data(show_spinner=False)
def load_all_rows():
    pdfs = find_pdf_paths()
    rows = []
    for p in pdfs:
        fname = os.path.basename(p).lower()
        station = "Kempegowda Bus Station (KBS)" if ("kbs" in fname or "kempegowda" in fname) else "Mysuru Road Bus Station (Satellite Bus Station)"
        try:
            parsed = parse_pdf_to_chunks(p, station)
            rows.extend(parsed)
        except Exception as e:
            logging.exception("Failed to parse %s: %s", p, e)
    return rows

st.title("KARNA - The KSRTC Assistant")
st.write("NAMASKARA! Welcome to KARNA, the Karnataka Automated Route Network Assistant! Whether you're chasing sunrise in Bengaluru or sunsets in Mysuru, I've got the latest KSRTC Bus schedules at your fingertips. Choose response mode, station, destination and service types to get your Bus timings.")

# Response mode prominently in main UI (Ensures visibility)
response_mode = st.radio("Response mode", ["Concise", "Detailed"], index=0, horizontal=True)

# Sidebar options (secondary)
with st.sidebar:
    st.header("Advanced Options")
    use_web_search = st.checkbox("Allow web search when no local match", value=True)
    use_fuzzy = st.checkbox("Use fuzzy matching", value=True)
    fuzzy_threshold = st.slider("Fuzzy threshold", 50, 100, 75)

# Station selection
station_choice = st.selectbox("Select departure bus station", [
    "Kempegowda Bus Station (KBS)",
    "Mysuru Road Bus Station (Satellite Bus Station)"
])

rows = load_all_rows()
station_rows = [r for r in rows if r.get("station") == station_choice]

all_services = sorted({(r.get("service") or "").strip() for r in station_rows if r.get("service")})
all_services = [s for s in all_services if s]
service_choice = st.multiselect("Select Service Type(s)", all_services, default=[])

destination = st.text_input("Type your destination", "")

if st.button("Search"):
    if not destination.strip():
        st.warning("Please type destination.")
        st.stop()

    dest_q = destination.lower().strip()
    use_service_filter = True if service_choice else False

    exact_matches = []
    for r in station_rows:
        to_v = (r.get("to") or "").lower()
        via_v = (r.get("via") or "").lower()
        if dest_q in to_v or dest_q in via_v:
            if use_service_filter:
                if (r.get("service") or "").strip() in service_choice:
                    exact_matches.append(r)
            else:
                exact_matches.append(r)

    matches = exact_matches
    used_fuzzy = False

    if not matches and use_fuzzy:
        used_fuzzy = True
        candidates = []
        map_rows = {}
        for r in station_rows:
            to_val = (r.get("to") or "").strip()
            via_val = (r.get("via") or "").strip()
            cand = to_val if to_val else ""
            if via_val:
                cand += " | via " + via_val
            cand = cand.strip()
            if not cand:
                continue
            candidates.append(cand)
            map_rows.setdefault(cand, []).append(r)
        candidates = list(dict.fromkeys(candidates))
        fuzzy_results = process.extract(dest_q, candidates, scorer=fuzz.partial_ratio, limit=20)
        selected_candidates = [cand for cand, score, _ in fuzzy_results if score >= fuzzy_threshold]
        for cand in selected_candidates:
            for r in map_rows.get(cand, []):
                if use_service_filter:
                    if (r.get("service") or "").strip() in service_choice:
                        matches.append(r)
                else:
                    matches.append(r)

    if matches:
        seen = set()
        unique_matches = []
        for m in matches:
            key = (m.get("from"), m.get("to"), m.get("via"), m.get("service"), m.get("dep_time"))
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        if response_mode == "Concise":
            bullets = []
            for m in unique_matches:
                srv = m.get("service") or ""
                to_p = m.get("to") or ""
                dep = m.get("dep_time") or ""
                bullets.append(f"- {srv} Bus to {dest_q} — {dep}")
            st.success("Bus timings (Concise):")
            st.markdown("\n".join(bullets))
        else:
            intro = f"The Karnataka State Road Transport Corporation has exemplary services connecting various cities within the state and other states. Here are the matching bus services from **{station_choice}** to **{destination.title()}**:"
            lines = []
            for m in unique_matches:
                srv = m.get("service") or ""
                to_p = m.get("to") or ""
                via_p = m.get("via") or ""
                dep = m.get("dep_time") or ""
                if via_p:
                    lines.append(f"- {srv} Bus to {to_p} via {via_p} at {dep}. This Bus goes through {dest_q}")
                else:
                    lines.append(f"- {srv} to {to_p} at {dep}")
            st.success("Bus timings (Detailed):")
            st.write(intro)
            st.markdown("\n".join(lines))

    else:
        st.warning("No matching rows found in local PDFs.")
        if use_web_search:
            snippets = serpapi_search(destination, num_results=3)
            if not snippets:
                st.error("Web search returned no useful results.")
            else:
                retrieved_docs = []
                for s in snippets:
                    retrieved_docs.append(f"{s.get('title')}\n{s.get('snippet')}\n{s.get('link')}")
                mode_for_llm = "concise" if response_mode == "Concise" else "detailed"
                prompt_text = (
                    f"Provide KSRTC bus information for Bengaluru to {destination}. "
                    f"If concise, give short bullet points with destination and departure time. "
                    f"If detailed, give a short paragraph introduction and then list timings with service and via."
                )
                answer = ask_llm(SYSTEM_PROMPT, retrieved_docs, prompt_text, mode=mode_for_llm)
                st.markdown(answer)
        else:
            st.info("Web search disabled. Enable it from sidebar.")
            
