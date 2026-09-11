import streamlit as st
import requests


st.title("🤖 Company Policy AI Assistant")

st.write(
    "Ask questions about company policies."
)


st.sidebar.title("About")

st.sidebar.write(
    "This application uses FastAPI and an LLM "
    "to answer user questions."
)

st.sidebar.write("Model: GPT-5 mini")


question = st.text_input(
    "Enter your question:"
)


if st.button("Ask"):

    if question:

        try:

            response = requests.get(
                "http://127.0.0.1:8000/ask",
                params={"question": question}
            )

            response.raise_for_status()

            data = response.json()

            st.subheader("Answer")

            st.write(data["answer"])

        except requests.exceptions.RequestException:

            st.error(
                "Could not connect to the FastAPI backend."
            )

    else:

        st.warning(
            "Please enter a question."
        )
