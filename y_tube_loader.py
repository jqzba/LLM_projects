import os
import textwrap

from apikey import apikey

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
##from langchain.memory import ConversationBufferMemory
##from langchain.utilities import WikipediaAPIWrapper 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
##from dotenv import find_dotenv, load_dotenv


##load_dotenv(find_dotenv())
os.environ['OPENAI_API_KEY'] = apikey



embeddings = OpenAIEmbeddings()

#Download multiple youtube transcripts TODO
def create_db_from_youtube_videos(video_urls): 
    all_transcripts = []
    
    for video_url in video_urls:
        try:
            loader = YoutubeLoader.from_youtube_url(video_url)
            transcript = loader.load()
            
            # Split the transcript into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(transcript)
            
            # Append each chunk to all_transcripts
            all_transcripts.extend(docs)
                
        except Exception as e:
            print(f"Failed to fetch transcript for video: {video_url}")
            print(e)
            continue

    # Use FAISS to store text as vectors
    db = FAISS.from_documents(all_transcripts, embeddings)  # You need to define 'embeddings'
    return db

#download one transcript and make FAISS
def create_db_from_url (video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    return db
    
def get_response_from_query(db, query, k=4):
    '''
    text-davinci-003 can handle up to 4096 tokens. So by setting up chunksize to 1000
    max the anlyzed tokens
    '''
    ##FAISS.similarity_search()
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    #choose the LLM 
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
         You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.""", 
    )
    
    #Human quetion prompt
    
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    
    return response, docs
    
    
# Example usage:
##video_url = "https://www.youtube.com/watch?v=zulGMYg0v6U"
##database = create_db_from_url(video_url)

# Example usage:
video_urls = ["https://www.youtube.com/watch?v=zulGMYg0v6U", "https://www.youtube.com/watch?v=N_P4pJjVTKc"]
database = create_db_from_youtube_videos(video_urls)


query = "How to build app with Chatgpt"
response, docs = get_response_from_query(database, query)
print(textwrap.fill(response, width=85))
      