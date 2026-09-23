import streamlit as st
import requests


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Company Policy AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title(
    "🤖 Company Policy AI Assistant"
)

st.write(
    "Ask questions about company policies "
    "and get AI-powered answers."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title(
    "About"
)

st.sidebar.write(
    "This application uses Retrieval-Augmented "
    "Generation (RAG) to answer questions "
    "using company policy documents."
)

st.sidebar.write(
    "Technologies:"
)

st.sidebar.write(
    "• Python"
)

st.sidebar.write(
    "• FastAPI"
)

st.sidebar.write(
    "• FAISS"

)

st.sidebar.write(
    "• Sentence Transformers"
)

st.sidebar.write(
    "• OpenAI"
)

st.sidebar.write(
    "• Streamlit"
)


# ==========================================
# EXAMPLE QUESTIONS
# ==========================================

st.subheader(
    "💡 Example Questions"
)

st.write(
    "• How many vacation days do employees receive?"
)

st.write(
    "• Do employees receive dental and vision insurance?"
)

st.write(
    "• Can employees work remotely?"
)

st.write(
    "• Are company laptops provided?"
)


# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_input(
    "Enter your question:"
)


# ==========================================
# ASK BUTTON
# ==========================================

if st.button(
    "Ask AI 🤖"
):

    if question:

        try:

            response = requests.get(
                "http://127.0.0.1:8000/ask",
                params={
                    "question": question
                },
                timeout=60
            )


            response.raise_for_status()


            data = response.json()


            st.subheader(
                "🤖 AI Answer"
            )


            st.success(
                data["answer"]
            )


        except requests.exceptions.RequestException:

            st.error(
                "Could not connect to the FastAPI backend. "
                "Please make sure FastAPI is running."
            )


    else:

        st.warning(
            "Please enter a question."
        )