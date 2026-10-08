# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()

# model=ChatGroq(model="openai/gpt-oss-120b")
# loader=PyPDFLoader("sample_summarization_document.pdf")

# data=loader.load()
# splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
# docs=splitter.split_documents(data)

# initial_prompt=PromptTemplate(template="Write a Detailed suumary of the following text:\n{text}",

# input_variables=['text'])
# refine_prompt=PromptTemplate(template='''Your current summary is :{exisiting_answer} now refine and improve it using the following new text:{text}
# provide an updated ,improved summary that includes important details from both the old and new text''',input_variables=['exisiting_answer','text'])

# initial_chain = initial_prompt | model
# refine_chain = refine_prompt | model


# response = refine_chain.invoke({  "text": docs[0].page_content})


# print(response.content)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Load PDF
loader = PyPDFLoader("sample_summarization_document.pdf")
data = loader.load()

# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

docs = splitter.split_documents(data)

# Initial prompt
initial_prompt = PromptTemplate(
    template="""Write a detailed summary of the following text:

{text}
""",
    input_variables=["text"]
)

# Refine prompt
refine_prompt = PromptTemplate(
    template="""Your current summary is:

{exisiting_answer}

Now refine and improve it using the following new text:

{text}

Provide an updated and improved summary that includes important
details from both the old and new text.
""",
    input_variables=["exisiting_answer", "text"]
)

# Chains
initial_chain = initial_prompt | model
refine_chain = refine_prompt | model

# Step 1: Create initial summary
response = initial_chain.invoke({
    "text": docs[0].page_content
})

summary = response.content

# Step 2: Refine summary using remaining chunks
for doc in docs[1:]:

    response = refine_chain.invoke({
        "exisiting_answer": summary,
        "text": doc.page_content
    })

    summary = response.content

# Final result
print("\n========== FINAL SUMMARY ==========\n")
print(summary)