import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
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
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k":3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("QA chain done")

    def ask_question(self, question):
        """
        ask a question to the bot

        args: question(str): question about uh buildings
        returns: dict: answer and source information
        """

        if not self.qa_chain:
            return {
                "error": "Bot did not initialize properly"
            }
        
        try:
            result = self.qa_chain({"query": question})

            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        
        except Exception as e:
            return {
                "error": f"Error processing question: {str(e)}"
            }
        
    def load_existing_vectorstore(self):
        """
        load previously saved vector store
        """
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model = "models/embedding-001",
                google_api_key=self.google_api_key
            )

            self.vectorstore = FAISS.load_local(
                "vectorstore/uh_buildings",
                self.embeddings
            )

            print("Loaded existing vector store")
            return True
        
        except:
            print("No existing vector store found")
            return False
        

def main():
    """
    main function to run the bot
    """

    print("UH Campus Guide Bot")
    print("=" * 40)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API key here: ")
    
    try:
        bot = UHCampusGuideBot(
            pdf_path="data/buildings.pdf",
            google_api_key=api_key
        )

        print("Bot is ready, Ask me about UH buildings")
        print("Type 'quit' to exit \n")

        while True:
            question = input("You: ").strip()

            if question.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye")
                break

            if not question:
                continue

            print("Thinking..")
            result = bot.ask_question(question)

            if "error" in result:
                print(f"Bot: Sorry, {result['error']}")
            else:
                print(f"Bot: {result['answer']}")

                if len(result['source_documents']) > 0:
                    print(f"\n Source: {len(result['source_documents'])} document(s) referenced")

            print("-"*40)
        
    except Exception as e:
        print(f"Error initializing bot: {e}")
        print("Check for buildings file location")

if __name__ == "__main__":
    main()
