# Import necessary libraries
from dotenv import find_dotenv, load_dotenv
# from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents import load_tools

from langchain_core.output_parsers import StrOutputParser
load_dotenv(find_dotenv())


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
            # tools = load_tools([ 'llm-math'], llm=llm)
            # prompt_template_name = PromptTemplate(input_variables=['DataFrame1', 'DataFrame2', 'DataFrame3','Year'], template="Find the difference between the Ending Balance in {DataFrame2} and sum of Beginning Balance in {DataFrame2} and USD amount from {DataFrame1}for the {Year} based on Period_Ending after joining them on Account Number only. Generate a  dataframe with account number, Beginning balance, USD Amount and Ending Balance where the difference generated earlier is not equal to 0. Show the resulting DataFrame."
# ) 
template="Find the difference between the Ending Balance in tb and sum of Beginning Balance in tb and USD amount from je for the {Year} based on Period_Ending after joining them on Account Number only. Generate a  dataframe with account number, Beginning balance, USD Amount and Ending Balance where the difference generated earlier is not equal to 0. Show the resulting DataFrame."
prompt_template_name = PromptTemplate(input_variables=['Year'], template=template
)
print(prompt_template_name.format( Year="2021"))    