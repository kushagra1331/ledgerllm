from dotenv import find_dotenv, load_dotenv
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
load_dotenv(find_dotenv())

# Sample DataFrames
df1 = pd.DataFrame({'account_number': [1, 2, 3], 'Balance1': [100, 200, 300]})
df2 = pd.DataFrame({'account_number': [2, 3, 4], 'Balance2': [150, 250, 350]})
df3 = pd.DataFrame({'account_number': [1, 3, 4], 'Balance3': [120, 220, 320]})

# Joining DataFrames
df_merged = pd.merge(df1, df2, on='account_number', how='outer')
df_merged = pd.merge(df_merged, df3, on='account_number', how='outer')

# LangChain - OpenAI setup
llmt = OpenAI(temperature=0)
prompt_template_name = PromptTemplate(
    input_variables=['DataFrame1', 'DataFrame2', 'DataFrame3'],
    template="Join the given DataFrames on 'account_number' and create a tabular output. Show the resulting DataFrame."
)
agent = create_pandas_dataframe_agent(llm=llmt, df=[df1, df2, df3], prompt=prompt_template_name, verbose=True)
response = agent.run(input={'DataFrame1': df1, 'DataFrame2': df2, 'DataFrame3': df3})  # Provide input as a dictionary

# Convert LangChain's response to a DataFrame
df_result = pd.read_json(response, orient='records')

# Display the resulting DataFrame
print(df_result)