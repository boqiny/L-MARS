"""Minimal single-turn Streamlit app wrapper."""
from __future__ import annotations

import streamlit as st

from lmars.workflow import create_workflow


def main() -> None:
    st.set_page_config(page_title="L-MARS", layout="wide")
    st.title("L-MARS Single-Turn Legal QA")

    model = st.text_input("Model", value="openai:gpt-4o-mini")
    use_cache = st.checkbox("Use cache", value=True)
    cache_dir = st.text_input("Cache dir", value="eval/cache")
    example_id = st.text_input("Example id", value="streamlit")
    question = st.text_area("Question")

    if st.button("Run") and question.strip():
        wf = create_workflow(mode="simple", llm_model=model)
        result = wf.run(
            query=question,
            example_id=example_id,
            use_cache=use_cache,
            cache_dir=cache_dir,
            seed=7,
        )
        st.subheader("Prediction")
        st.write(result["final_answer"])
        st.subheader("Retrieved Evidence")
        st.json(result["retrieved"])


if __name__ == "__main__":
    main()
