import os


from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap

from apikey import apikey, hugging_face

# --------------------------------------------------------------
# Load the HuggingFaceHub API token 
# --------------------------------------------------------------

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face

# --------------------------------------------------------------
# Load the LLM model from the HuggingFaceHub
# --------------------------------------------------------------

repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)


#Create a PromptTemplate
template = """Question: {question}
Answer: Let's think step by step"""

prompt = PromptTemplate(template= template, input_variables = ["question"])
llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)


#Run the LLMChain
question = "How do I become indfluencer"
response = llm_chain.run(question)
wrapped_text = textwrap.fill(response, width=200, break_long_words=False, 
                             replace_whitespace=False)
print(wrapped_text)



#lets test with the video transcript
video_url = "https://www.youtube.com/watch?v=hdjCm-vxrBI"
loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
docs = text_splitter.split_documents(transcript)



#Summarization with LangChain
chain = load_summarize_chain(falcon_llm, chain_type="map_reduce", verbose=True)
print(chain.llm_chain.prompt.template)
print(chain.combine_document_chain.llm_chain.prompt.template)


#Test the Falcon model with text summarization
output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=200, 
                             break_long_words=False, replace_whitespace=False)
print(wrapped_text)






