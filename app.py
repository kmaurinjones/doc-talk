import os
import time
import streamlit as st
import pdfplumber
import docx
from nltk.tokenize import sent_tokenize
import tiktoken
from openai import OpenAI
import numpy as np

# Streamlit interface setup
st.set_page_config(
    page_title="DocTalk 🤖📚",
    layout="wide",
    page_icon="🤖📚 ",
    initial_sidebar_state="collapsed"
)

# setting sidebar
st.sidebar.title("Response Contexts")
with st.sidebar:
    st.divider() # just makes it look nicer

# default val
st.session_state.needs_context = False

# App title
st.title("DocTalk 🤖📚")
st.markdown("*Chat with Your Documents!*")

# Initialize OpenAI API
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    info = {}
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            text += page_text
            info[page.page_number] = page_text
    return text, info

# Function to extract text from Word document
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    info = {}
    for i, para in enumerate(doc.paragraphs):
        text += para.text
        info[i + 1] = para.text
    return text, info

# Function to process uploaded files
def process_files(files):
    texts = {}
    for file in files:
        if file.name not in st.session_state.parsed_files:
            if file.type == "application/pdf":
                texts[file.name] = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                texts[file.name] = extract_text_from_docx(file)
            elif file.type == "text/plain":
                text = file.read().decode("utf-8")
                texts[file.name] = (text, {1: text})
    return texts

# Function to chunk text with overlapping sentences
def chunk_text(text, file_name, page_info, max_tokens=350, overlap_sentences=2):
    enc = tiktoken.encoding_for_model("gpt-4o")
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    
    for i, sentence in enumerate(sentences):
        tokenized_sentence = enc.encode(sentence)
        sentence_length = len(tokenized_sentence)
        
        if current_chunk_length + sentence_length > max_tokens:
            chunks.append((file_name, page_info.get(i + 1, "No page number available for this context"), " ".join(current_chunk)))
            current_chunk = current_chunk[-overlap_sentences:]
            current_chunk_length = sum(len(enc.encode(s)) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_chunk_length += sentence_length
    
    if current_chunk:
        chunks.append((file_name, page_info.get(len(sentences), "No page number available for this context"), " ".join(current_chunk)))
    
    return chunks

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Initialize session state variables
if 'parsed_files' not in st.session_state:
    st.session_state.parsed_files = {}
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_log = []

uploaded_files = st.file_uploader("Upload your documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.write("Once you have uploaded all documents, click the button below to process them.")

    if st.button("Process Documents"):
        new_texts = process_files(uploaded_files)
        
        ### put this within the expander below ###
        if new_texts:
            for file_name, (text, info) in new_texts.items():
                st.session_state.parsed_files[file_name] = (text, info)

        for file_name, (text, info) in st.session_state.parsed_files.items():
            if file_name not in st.session_state.embeddings:
                with st.expander(f"Processing {file_name}"):
                    progress_bar = st.progress(0)
                    chunks = chunk_text(text, file_name, info)
                    embeddings = []
                    for i, chunk in enumerate(chunks):
                        response = client.embeddings.create(input=chunk[2], model="text-embedding-3-small")
                        embeddings.append(response.data[0].embedding)
                        progress_bar.progress((i + 1) / len(chunks))
                    st.session_state.embeddings[file_name] = (chunks, embeddings)
            st.success(f"{file_name} processed.")

def query_eval(query, document_titles):
    """Helper function GPT-4o API call that returns True if query is looking for document context and False if not."""
    classify_query_base_prompt = """

    This user query exists in the context of a RAG-style 'Chat with your documents' app.

    # Instruction
    Classify whether this user query is asking for information from the uploaded documents.
    Keep in mind that it is possible that the user is asking for context from the rest of the conversation, but not further document-provided context.
    Default to 1 (yes) if unsure.
    Return exactly [1, 2] to resresent [Yes, No] respectively.

    # User Query
    "{user_query}"

    # Sample Response
    1

    # Sample Response
    2
    """.replace("\t", "").strip()

    # response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": classify_query_base_prompt.format(document_titles=document_titles, user_query=query)}])
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": classify_query_base_prompt.format(user_query=query)}])
    
    return response.choices[0].message.content.strip() == "1"

def stream_response(response):
    for chunk in response:
        time.sleep(0.04) # artificial delay to make it look cooler
        # ignore None chunks
        if chunk or chunk == " ":
            yield chunk.choices[0].delta.content

# Chat input and message display
if "embeddings" in st.session_state and st.session_state.embeddings:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_context" not in st.session_state:
        st.session_state.show_context = False

    # Display the messages
    for message in st.session_state.messages:
        if message['content'] and message['content'].strip():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # If the user types something in the chat box
    if prompt := st.chat_input("Ask a question about your documents:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Validate query
            needs_context = query_eval(prompt, st.session_state.parsed_files.keys())
            st.session_state.needs_context = needs_context

            messages = [{"role": "system", "content": "You are a helpful assistant that helps users get information from their documents."}]
            for message in st.session_state.messages:
                messages.append({"role": message["role"], "content": message["content"]})

            print(f"Context needed: {'TRUE' if needs_context else 'FALSE'}") # print to console for debugging

            if needs_context:
                # Get context for the query
                query_embedding = client.embeddings.create(input=prompt, model="text-embedding-3-small").data[0].embedding
                similarities = []
                results_info = []
                for file_name, (chunks, embeddings) in st.session_state.embeddings.items():
                    for i, emb in enumerate(embeddings):
                        sim = cosine_similarity(query_embedding, emb)
                        similarities.append((sim, chunks[i]))
                        results_info.append((sim, file_name, chunks[i][1], chunks[i][2]))

                top_k_chunks = [chunk for _, chunk in sorted(similarities, key=lambda x: x[0], reverse=True)[:3]]
                top_k_info = [(file_name, page, chunk) for _, file_name, page, chunk in sorted(results_info, key=lambda x: x[0], reverse=True)[:5]]

                chunks_str = ""
                for chunk in top_k_chunks:
                    chunks_str += f"{chunk[2]}\n\n"

                base_user_prompt = """
                # Instruction
                Respond to the user's query given the context of the uploaded documents and relevant conversation history.
                Only draw from the most relevant information of all immediate and long-term context to inform your response.
                If you are unsure or need more information, ask the user for clarification, or say that you are unsure.
                Do not provide false information. If you do not know the answer, say that you do not know.
                You meaningfully summarize large amounts of information.
                
                # Provided Context
                {context}

                # User Query
                {query}
                """.strip()

                messages.append({"role": "user", "content": base_user_prompt.format(context=chunks_str.strip(), query=prompt)})
            else:
                base_user_prompt = """
                # Instruction
                Only draw from the most relevant information from the context of this conversation to inform your response.
                If you are unsure or need more information, ask the user for clarification, or say that you are unsure.
                Do not provide false information. If you do not know the answer, say that you do not know.

                # User Query
                {query}
                """.strip()

                messages.append({"role": "user", "content": base_user_prompt.format(query=prompt)})

            # Generate response
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True
            )

            for chunk in stream_response(stream):
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            # Clear the placeholder and write the final response
            response_placeholder.empty()
            st.markdown(full_response)

            # Add response info if context was used
            if needs_context:
                response_info = ""
                for info in top_k_info:
                    response_info += f"**- File:** {info[0]}\n"
                    response_info += f"**- Page:** {info[1]}\n"
                    context = info[2][:100].replace('\n', ' ')
                    response_info += f'**- Content:** "...{context}..."\n'
                    response_info += "\n\n\n"

                if response_info.strip():
                    st.session_state.messages.append({"role": "assistant", "content": full_response, "context": response_info.strip()})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Only show "context" button if there is at least one model turn of conversation
    if st.session_state.messages:
        if st.session_state.needs_context:
            with st.container():
                st.markdown("\t📂 *There is context available for this response! Open the sidebar to see it.*")

        with st.sidebar:
            for idx, message in enumerate(st.session_state.messages):
                if "context" in message and message["role"] == "assistant":
                    st.markdown(f"## **User Query:**\n{st.session_state.messages[idx-1]['content']}\n\n")
                    st.markdown(f"## **Context:**\n{message['context']}\n\n")
                    st.divider()
