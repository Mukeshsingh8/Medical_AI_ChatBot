# Import necessary modules and define environment variables
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain, LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Text splitter and system template
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """You are a friendly doctor conversing with a patient. Given the uploaded medical report, please answer the patient's questions accurately using the information from the report. If needed, supplement your answers with general medical knowledge to provide a comprehensive response. Ensure your responses are clear, empathetic, and easy to understand."""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

@cl.on_chat_start
async def on_chat_start():
    elements = [
        cl.Image(name="image1", display="inline", path="./robot.png")
    ]
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to your medical reports! I'm here to help you understand your report better. Please upload your report to get started.", elements=elements).send()
    files = None

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file using the path attribute
    with open(file.path, 'rb') as f:
        pdf_content = f.read()

    pdf_stream = BytesIO(pdf_content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Create the LLM
    llm = ChatOpenAI(temperature=0.5, streaming=True)

    # Create the LLM chain for combining documents
    document_combination_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the following context: {context}\nAnswer the following question: {question}"
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=document_combination_prompt
    )
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    # Define the question generation chain
    question_generation_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="Combine the chat history and follow-up question into a standalone question.\nChat History: {chat_history}\nFollow-up Question: {question}"
    )
    question_generator_chain = LLMChain(
        llm=llm,
        prompt=question_generation_prompt
    )

    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator_chain,
        return_source_documents=True
    )

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions about your report!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    text = message.content  # Extract text content from the Message object
    if not text.strip():  # Check if the text is not just empty or whitespace
        await cl.Message(content="Invalid input. Please provide a valid query.").send()
        return

    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # Pass the extracted text to the chain method
    res = await chain.ainvoke({"question": text, "chat_history": []}, callbacks=[cb])

    answer = res.get("answer", "I'm sorry, I couldn't find an answer to your question.")
    sources = res.get("sources", "").strip()
    source_elements = []

    # Process sources and prepare response
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
