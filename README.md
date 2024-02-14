# RAG chatbot powered by 🔗 Langchain, OpenAI, Google Generative AI and Hugging Face 🤗

<div align="center">
  <img src="https://github.com/AlaGrine/RAG_chatabot_with_Langchain/blob/main/data/docs/RAG_architecture.png" >
  <figcaption>RAG architecture with Langchain components.</figcaption>
</div>

## Project Overview <a name="overview"></a>

Although Large Language Models (LLMs) are powerful and capable of generating creative content, they can produce outdated or incorrect information as they are trained on static data. To overcome this limitation, Retrieval Augmented Generation (RAG) systems can be used to connect the LLM to external data and obtain more reliable answers.

The aim of this project is to build a RAG chatbot in Langchain powered by [OpenAI](https://platform.openai.com/overview), [Google Generative AI](https://ai.google.dev/?hl=en) and [Hugging Face](https://huggingface.co/) **APIs**. You can upload documents in txt, pdf, CSV, or docx formats and chat with your data. Relevant documents will be retrieved and sent to the LLM along with your follow-up questions for accurate answers.

Throughout this project, we examined each component of the RAG system from document loader to conversational retrieval chain. Additionally, we developed a user interface using [streamlit](https://streamlit.io/) application.

## Installation <a name="installation"></a>

This project requires Python 3 and the following Python libraries installed:

`langchain` ,`langchain-openai`, `langchain-google-genai`, `chromadb`, `streamlit`, `streamlit`

The full list of requirements can be found in `requirements.txt`

## Instructions <a name="instructions"></a>

To run the app locally:

1. Create a virtual environment: `python -m venv langchain_env`
2. Activate the virtual environment : `.\langchainenv\Scripts\activate` on Windows.
3. Run the following command in the directory: `cd RAG_Chatabot_Langchain`
4. Install the required dependencies `pip install -r requirements.txt`
5. Start the app: `streamlit run RAG_app.py`
6. In the sidebar, select the LLM provider (OpenAI, Google Generative AI or HuggingFace), choose an LLM (GPT-3.5, GPT-4, Gemini-pro or Mistral-7B-Instruct-v0.2), adjust its parameters, and insert your API keys.
7. Create or load a Chroma vectorstore.
8. Chat with your documents: ask questions and get 🤖 AI answers.

## Blog post <a name="blog_post"></a>

I wrote a blog post about this project. You can find it [here](https://medium.com/@alaeddine.grine/rag-chatbot-powered-by-langchain-openai-google-generative-ai-and-hugging-face-apis-6a9b9d7d59db)
