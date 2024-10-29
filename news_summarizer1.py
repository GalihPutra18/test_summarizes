import spacy
import streamlit as st

try:
    nlp_en = spacy.load('en_core_web_sm')
    nlp_id = spacy.load('xx_ent_wiki_sm')
    st.success("Models loaded successfully!")
except OSError as e:
    st.error(f"Failed to load SpaCy models: {e}")
