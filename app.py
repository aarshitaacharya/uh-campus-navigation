import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
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
    
    def create_vector_store(self, documents):
        """
        create embeddings and vector stores
        
        args: documents (list): list of document chunks
        """

        print("Creating embeddings and vector store")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model = "models/embedding-001",
            google_api_key = self.google_api_key
        )

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        self.vectorstore.save_local("vectorstore/uh_buildings")
        print("Vector store created and saved")

    def setup_qa_chain(self):
        """
        Set up the chain with custom prompt
        """
        print("QA chain started")
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key = self.google_api_key,
            temperature="0.1"
        )

        prompt_template = """
        You are a helpful University of Houston (UH) campus guide assistant. 
        Use the following context about UH buildings to answer questions accurately.
        
        Context: {context}
        
        Question: {question}
        
        Instructions:
        - Provide accurate information about UH buildings based on the context
        - If asked about building codes, abbreviations, or names, use the exact information from the context
        - If you don't know something, say you don't have that information
        - Be helpful and friendly like a campus guide would be
        - Include building codes/abbreviations when relevant
        
        Answer:
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriver = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k":3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("QA chain done")
