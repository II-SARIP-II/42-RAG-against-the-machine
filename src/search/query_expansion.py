import dspy


class Expansion_sign(dspy.Signature):
    """Generate synonyms from the word of the user's query"""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="A single short line question of alternative keywords, acronyms, and synonyms. No bullet points, no markdown."
    )
