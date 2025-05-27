
prompt_template = """You are an expert in summarizing technical texts for a Retrieval Augmented Generation system.
Given the following retrieved text chunks, provide a concise and factual abstractive summary.
The summary should synthesize the information and highlight the most important points relevant to building efficient context for an LLM.
Do not add any information not present in the text.

Retrieved Text:
---
{text_to_summarize}
---

Concise Abstractive Summary:
"""