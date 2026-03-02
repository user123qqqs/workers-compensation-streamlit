import streamlit as st

analysis_page = st.Page("analysis_and_model.py", title="Анализ и модель")
presentation_page = st.Page("presentation.py", title="Презентация")

pages = {
    "Разделы": [analysis_page, presentation_page],   
}

current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()