import dspy


class SingleSentenceAnswer(dspy.Signature):
    """ONLY Answer the question using a single, short,
    and direct sentence based ONLY on the provided context.
    Do not include introductory text, reasoning, or labels like 'Answer:'."""

    context = dspy.InputField(desc="Extracted chunks from the codebase.")
    question = dspy.InputField(desc="The technical question to answer.")
    answer = dspy.OutputField(
        desc="A single direct sentence answering the question, "
        "or 'Information missing.' if not found.")
