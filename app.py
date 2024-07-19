import os
import time
import streamlit as st
import pdfplumber
import docx
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import tiktoken
from openai import OpenAI
import numpy as np

# MODEL = "gpt-4o"
MODEL = "gpt-4o-mini-2024-07-18"

# Streamlit interface setup
st.set_page_config(
    page_title="DocTalk 🤖📚",
    layout="wide",
    page_icon="🤖📚 ",
    initial_sidebar_state="collapsed",
)

# Define the correct passcode
correct_passcode = os.getenv("SIMPLE_AUTH_PASSCODE")

# Function to display the passcode input box
def display_passcode_prompt():
    st.title("Access Required")
    st.write("Please enter the passcode to access the app.")
    passcode = st.text_input("Passcode", type="password", label_visibility="hidden")
    if st.button("Submit"):
        if passcode == correct_passcode:
            st.session_state.access_granted = True
            st.experimental_rerun()
        else:
            st.error("Incorrect passcode. Please try again.")

# Initialize session state variable
if 'access_granted' not in st.session_state:
    st.session_state.access_granted = False

# Display the passcode prompt if access is not granted
if not st.session_state.access_granted:
    display_passcode_prompt()
else:
    # setting sidebar
    st.sidebar.title("Response Contexts")
    with st.sidebar:
        st.divider() # just makes it look nicer

    # default val
    st.session_state.needs_context = False

    # top k chunks to consider
    top_k = 5

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
                print(page.page_number)
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
    def chunk_text(text, file_name, page_info, max_tokens=500, overlap_sentences=2):
        enc = tiktoken.encoding_for_model("gpt-4o")
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_chunk_length = 0
        current_page_number = 1

        for i, sentence in enumerate(sentences):
            tokenized_sentence = enc.encode(sentence)
            sentence_length = len(tokenized_sentence)

            if current_chunk_length + sentence_length > max_tokens:
                chunks.append((file_name, current_page_number, " ".join(current_chunk)))
                current_chunk = current_chunk[-overlap_sentences:]
                current_chunk_length = sum(len(enc.encode(s)) for s in current_chunk)

            current_chunk.append(sentence)
            current_chunk_length += sentence_length
            
            # Update page number based on the sentence index in the original text
            for page, content in page_info.items():
                if sentence in content:
                    current_page_number = page
                    break

        if current_chunk:
            chunks.append((file_name, current_page_number, " ".join(current_chunk)))
        
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
                            # response = client.embeddings.create(input=chunk[2], model="text-embedding-3-large")
                            embeddings.append(response.data[0].embedding)
                            progress_bar.progress((i + 1) / len(chunks))
                        st.session_state.embeddings[file_name] = (chunks, embeddings)
                st.success(f"{file_name} processed.")

    def query_eval(query, document_titles, model=MODEL):
        """Helper function for returning MODEL API call that returns True if query is looking for document context and False if not."""
        classify_query_base_prompt = """

        This user query exists in the context of a RAG-style 'Chat with your documents' app.

        # Instruction
        Classify whether this user query is asking for information from the uploaded documents.
        Keep in mind that it is possible that the user is asking for context from the rest of the conversation, but not further document-provided context.
        Default to 1 (yes) if unsure.
        Return exactly [1, 2] to represent [Yes, No] respectively.

        # Here are the document file paths that the user has uploaded:
        {document_titles}

        # User Query
        "{user_query}"

        # Sample Response
        1

        # Sample Response
        2
        """.replace("\t", "").strip()

        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": classify_query_base_prompt.format(document_titles=document_titles, user_query=query)}])
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
                document_titles = ", ".join(st.session_state.parsed_files.keys())
                needs_context = query_eval(prompt, document_titles)
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

                    # `top_k` defined at beginning of script
                    top_k_chunks = [chunk for _, chunk in sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]]
                    top_k_info = [(sim, file_name, page, chunk) for sim, file_name, page, chunk in sorted(results_info, key=lambda x: x[0], reverse=True)[:5]]

                    chunks_str = ""
                    for chunk in top_k_chunks:
                        chunks_str += f"{chunk[2]}\n\n"

                    base_user_prompt = """
                    # Instruction
                    Respond to the user's query given the context of the uploaded documents and relevant conversation history.
                    Only draw from the most relevant information of all immediate and long-term context to inform your response.
                    If you are unsure or need more information, ask the user for clarification, or say that you are unsure.
                    Do not provide false information. If you do not know the answer, say that you do not know.
                    You meaningfully summarize large amounts of information into as brief a response as possible.
                    You are concise and to the point, without skipping important details. You do NOT write summary conclusion statements at the end of your responses.
                    You leverage Markdown formatting to make your response more readable and structured, using bullet points, lists, and other formatting options where appropriate. No code blocks.
                    
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
                    model=MODEL,
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

                    # re-sort `top_k_info` by page number -- this is purely for display purposes
                    top_k_info = sorted(top_k_info, key=lambda x: x[2])

                    for info in top_k_info:
                        response_info += f"**- Similarity to Query:** {round(info[0] * 100, 2)}%||"
                        response_info += f"**- File:** {info[1]}||"
                        response_info += f"**- Page:** {info[2]}||"
                        context = info[3][:100].replace('\n', ' ')
                        response_info += f'**- Content:** "...{context}..."||'
                        response_info += ":::"

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
                        st.markdown('\t📂 *There is context available for this response! Open the sidebar using the **arrow icon** in the top left corner of this page to see it.*')

                with st.sidebar:
                    for idx, message in enumerate(st.session_state.messages):
                        if "context" in message and message["role"] == "assistant":
                            query_excerpt = st.session_state.messages[idx - 1]['content']
                            preview_chars = 40

                            # Initialize an empty string for the query excerpt
                            query_header_split = query_excerpt.split(" ")
                            query_excerpt_preview = ""

                            # Iterate through the words to construct a preview up to the specified character limit
                            too_long = False
                            for word in query_header_split:
                                if len(query_excerpt_preview) + len(word) + 1 > preview_chars:
                                    too_long = True
                                    break
                                query_excerpt_preview += word + " "

                            # Strip any trailing whitespace from the preview
                            query_excerpt_preview = query_excerpt_preview.strip()

                            # Generate the markdown with the preview logic
                            query_header_md = f'"*{query_excerpt_preview[:preview_chars] + "..." if too_long else query_excerpt_preview}*"'

                            with st.expander(query_header_md):
                                st.divider()
                                full_context = message['context'].split(":::")
                                for idx, context in enumerate(full_context):
                                    if context.strip():
                                        context = context.split("||")
                                        for c in context:
                                            st.write(c.strip())

                                        if idx < len(full_context) - 1:
                                            st.divider() # to space out context segments
