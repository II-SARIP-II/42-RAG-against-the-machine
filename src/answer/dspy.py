import dspy


class RAG_sign(dspy.Signature):
    """Answer the question using only the provided sources."""
    context: str = dspy.InputField(
        desc="Numbered source chunks, most relevant first"
    )
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="answer ONLY. one or two sentences. Mention the original source. No markdown, No symbols."
    )
