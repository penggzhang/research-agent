import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
#import streamlit as st
from langchain.callbacks import get_openai_callback
from fastapi import FastAPI

load_dotenv()
browserless_api_key = os.getenv('BROWSERLESS_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')

# 1. Tools for searching
def search(query):
    """
    Use Serper's API to search websites given the query.
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)
    return response.text

# search('courses on ESG corporate practice')

# 2. Tools for scraping
def scrape_website(objective: str, url: str):
    """
    User Browserless's API to scrape a website. If the content is 
    too large for LLM's token limit, then summarize the content 
    based on the objective.
    - Objective is the original objective/task that user give to the agent. 
    - Url is the url of the website to scrape.
    """
    # Scrape website
    print('\nScraping website ...')

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
    }

    data = {
        'url': url
    }
    data_json = json.dumps(data)

    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"

    response = requests.post(post_url, headers=headers, data=data_json)

    # If scraping succeeds,
    if response.status_code == 200:
        # then extract content as text from the html.
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        # If text is too large, then summarize it over objective.
        if len(text) > 10000:
            output = summarize(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code: {response.status_code}")

def summarize(objective, content):
    # Split content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])

    # Prepare prompt template for summarizing
    summary_prompt = """
    Summarize the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    summary_prompt_template = PromptTemplate(
        template=summary_prompt, input_variables=["text", "objective"]
    )

    # Setup llm
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    # Load and run summarize_chain
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=summary_prompt_template,
        combine_prompt=summary_prompt_template,
        verbose=True
    )
    output = summary_chain.run(input_documents=docs, objective=objective)
    return output

# scraped_output = scrape_website('company names', 'https://www.technologyreview.com/2023/10/05/1080952/climate-tech-companies-to-watch/')
# print(scraped_output)

# Wrap my functions as langchain tools
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool()
]

system_message = SystemMessage(
    content="""
    You are a world class researcher, who can do detailed research on any topic and produce fact-based results. 
    You do not make things up. You will try as hard as possible to gather facts & data to back up the research.
            
    Please make sure you complete the objective above with the following rules:
    1/ You should do enough research to gather as much information as possible about the objective.
    2/ If there are url of relevant links & articles, you will scrape it to gather more information.
    3/ After scraping & search, you should think "is there any new things i should search & scrape based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins.
    4/ You should not make things up, you should only write facts & data that you have gathered.
    5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research.
    6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research.
    """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

memory = ConversationSummaryBufferMemory(
    memory_key="memory",
    return_messages=True,
    llm=llm,
    max_token_limit=1000
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs=agent_kwargs,
    memory=memory,
    verbose=True
)

# 4. Use streamlit to create a web app
# def main():
#     st.set_page_config(page_title="AI Research Agent", page_icon="🛻")
#     st.header("AI Research Agent")

#     query = st.text_input("Please specify your research topic:")
#     if query: 
#         st.write("Doing research for: ", query)

#         with get_openai_callback() as cb:            
#             result = agent({"input": query})
#             print(cb)
#             st.info(result['output'])

# if __name__ == '__main__':
#     main()

# 5. Make this app as an API via FastAPI
app = FastAPI()

class Query(BaseModel):
    query: str

@app.post('/')
def researchAgent(query: Query):
    query = query.query
    result = agent({'input': query})
    return result['output']
