import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai

class UHCampusGuideBot:
    def __init__(self, pdf_path = "data/buildings.pdf", google_api_key=None):
        """
        initialize the bot

        args: 
            pdf_path(str): path to the buildings pdf file
            google_api_key(str): google's api key for gemini
        """

        self.pdf_path = pdf_path
        self.google_api_key = google_api_key

        # initialize some components 
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None

        self.setup_bot()
    
    def setup_bot(self):
        """
        set up the components of our bot
        """
        print("Setting up UH Campus Guide Bot")
        
        genai.configure(api_key=self.google_api_key)
        documents = self.load_pdf()
        self.create_vector_store(documents)
        self.setup_qa_chain()

        print("Bot setup done")
    
    def load_pdf(self):
        """
        loads and splits the pdf

        returns a list of document chunks
        """

        print(f"Loading pdf from {self.pdf_path}..")

        loader = PyMuPDFLoader(self.pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "", " "]
        )

        docs = text_splitter.split_documents(documents)
        print(f"Loaded {len(docs)} document chunks")

        return docs