# Importing the necessary modules and classes
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

# Importing json module for JSON handling
import json

# Importing os module for operating system dependent functionality
import os

# Setting OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = "XYZ"

# provide the path of  pdf file
pdfreader = PdfReader('handbook.pdf')

# Initializing an empty string to store the extracted text from the PDF
raw_text = ''

# Looping through each page in the PDF
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text() # Extracting text from the current page
    if content: # If content exists (not empty)
        raw_text += content # Appending extracted content to the raw_text string

# Creating a CharacterTextSplitter instance
text_splitter = CharacterTextSplitter(
    separator = "\n", # Separator for splitting text
    chunk_size = 800, # Maximum size of each text chunk
    chunk_overlap  = 200, # Overlapping size between consecutive chunks
    length_function = len, # Function to calculate length of text
)

# Splitting the raw text into chunks using the text_splitter
texts = text_splitter.split_text(raw_text)

# Instantiate embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Creating a FAISS index from the extracted text chunks
document_search = FAISS.from_texts(texts, embeddings)

# Loading the question answering chain using OpenAI language model
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Define the list of questions
query = [
    "What is the name of the company?",
    "Who is the CEO of the company?",
    "What is the capital of India?",
    "Who is the father of computer science?",
    "Can I use the company provided credit card to buy movie tickets?",    
    "I see that the company sponsors travel. Can I reimburse my vacation to Italy?",
    "I am planning to gift a potential client but the cost of the gift is 26 USD. Will this be allowed?",
    "What type of entertainment expenses are allowed as per policy? If I take my friend to a theme park can I claim that expense?",
    "Should the employees speak to the CEO if they find any discrepancy in the  job description?",
    "Does the company have a job description for the individual jobs?",
    "Does the company accommodate and support pregnant employees?",
    "Who has the authority to make promises or negotiate with regard to guaranteed or continued employment?",
    "What type of business does the company do?",
    "What is 1 + 1?",
    "What is their vacation policy?",
    "What is the termination policy?"
    ]

# Initilaize Dictionary to store the results
results={}

# Iterating over each question in the query list
for query in query:
    docs = document_search.similarity_search(query) # Finding similar documents for the current question
    result = chain.invoke({'input_documents': docs, 'question': query}) # Invoking the question answering chain for the current question
    results[query] = result['output_text'] # Storing the question-answer pair in the results dictionary
    
output_json = json.dumps(results, indent=4)

# Printing the structured JSON blob
print(output_json)

#print(OpenAI())



