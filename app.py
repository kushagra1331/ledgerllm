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
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain.output_parsers import PandasDataFrameOutputParser
# from langchain_experimental.llms import LangchainBase
# from langchain_openai import la
import streamlit as st
from datetime import datetime
load_dotenv(find_dotenv())
# from custom_pandas_agent import create_pandas_dataframe_agent
# from custom_pandas_agent import CustomChatOpenAI
# from custom_pandas_agent import create_pandas_dataframe_agent_custom

import pandas as pd
# class LangchainChatOpenAI(OpenAI):

#     def __init__(self, model="gpt-3.5-turbo-0613", temperature=0):
#         super().__init__(model=model, temperature=temperature)

# Load environment variables

# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_core.output_parsers import BaseOutputParser




# Read sample data
df1 = pd.read_excel(r"C:\Users\iamku\Documents\Sample\JE.xlsx")
df2 = pd.read_excel(r"C:\Users\iamku\Documents\Sample\TB.xlsx")
df3 = pd.read_excel(r"C:\Users\iamku\Documents\Sample\COA.xlsx")

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# Set target column
target_column = 'posted date'
target_column_index = df1.columns.get_loc(target_column)
je = df1.iloc[:, :target_column_index + 1]

# Get date range
mindate = je['Period_Ending'].min()
maxdate = je['Period_Ending'].max()

from typing import Dict, Any
from langchain_core.tools import BaseTool

st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
# Streamlit app
def main():
    st.title("General Ledger Analyzer")

    # Sidebar with options
    option = st.sidebar.selectbox("Select Analysis Type", ["General Ledger Reconciliation", "Period Over Period Comparison"], index=None)

    # Main content based on user selection
    if option == "General Ledger Reconciliation":

        st.header("General Ledger Recon Analysis")

        # Get the last three years
        current_year = datetime.now().year
        years = list(range(df1['Period_Ending'].dt.year.min(), df1['Period_Ending'].dt.year.max() + 1),)
        selected_year = st.selectbox("Select Year Ending Date", years, index=None)
  

        st.write(f"Selected Year: {selected_year}")
        if selected_year:
            # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
            # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
            # tools = load_tools(llm=llm)
            # tools = load_tools([ repl_tool], llm=llm)
#             prompt_template_name = PromptTemplate(input_variables=['DataFrame1', 'DataFrame2', 'DataFrame3','Year'], template="Find the difference between the Ending Balance in {DataFrame2} and sum of Beginning Balance in {DataFrame2} and USD amount from {DataFrame1}for the {Year} based on Period_Ending after joining them on Account Number only. Generate a  dataframe with account number, Beginning balance, USD Amount and Ending Balance where the difference generated earlier is not equal to 0. Show the resulting DataFrame."
# ) 
            prompt_template_name = PromptTemplate(input_variables=['Year'], template="WHat is the difference between the je and tb Ending Balance  and sum of Beginning Balance  and USD amount for the {Year} based on Period_Ending.Ignore df3 for this  "
) 
            # prompt_template_name = PromptTemplate(input_variables=['Year'], template="Find the difference between the Ending Balance in tb and sum of Beginning Balance in tb and USD amount from je for the {Year} based on Period_Ending after joining them on Account Number only. Generate a  dataframe with account number, Beginning balance, USD Amount and Ending Balance where the difference generated earlier is not equal to 0. Show the resulting DataFrame."
# )
            # print(    prompt_template_name.format(        Year=selected_year   ))       
             # prompt_template_name = PromptTemplate(
        #     input_variables=['Year'],
        #     template="Find the difference between the Ending Balance in TB and sum of Beginning Balance and USD amount for the {Year} based on Period_Ending after joining them on Account Number. Generate a pandas dataframe with account number, Beginning balance, USD Amount and Ending Balance where the difference generated earlier is not equal to 0 "
        # )
        # agent = create_pandas_dataframe_agent(llm=llmt, df=[je, tb, coa], prompt=prompt_template_name, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
        # llm_dict = {'llm': llmt}
            # agent = create_pandas_dataframe_agent(llm, df=[je, tb, coa],model="gpt-3.5-turbo-0613", temperature=0, prompt=prompt_template_name, verbose=True,tools=BaseTool)
            # agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [je, tb,coa] ,prompt=prompt_template_name,verbose=True)
            agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0,model='gpt-4-0613'), [df1,df2,df3] ,verbose=True)
            # executor = AgentExecutor(agent=agent,handle_parsing_errors=True,tools=repl_tool)
            # response = agent.run('Filter  df1,df2 for the {Year} based on Period_Ending column , then Find the sum of Beginning_Balance in df2 and USD amount from df1  after joining them on Account Number only, then subtract the sum with Ending_Balance in df2 and show only account number,Beginning_Balance,USD_Amount, Ending_Balance and difference in a tabular format where difference is not 0 for the selected {year}')
            response1 = agent.run(f'Filter  df1,df2 for the {selected_year} based on Period_Ending column and join them based of Account Num')
            
            response2 = agent.run(f'In {response1} Find the sum of Beginning_Balance and USD amount and create a new column Total_Beginning and show the output dataset ')
            response3 = agent.run(f'In {response2} Subtract the Ending_Balance with Total_Beginning and create a new column Variance and show the output dataset ')
            response4 = agent.run(f'In {response3} Show only those rows where variance is not 0 and show the output dataset ')
            # response = agent.run("Total records in JE dataframe")
            # st.write("Prompt Passed:", prompt_template_name)
            
            # st.write("Raw Response:", response)
            st.write("Filter:", response1)
            st.write("Total Beginning:", response2)
            st.write("Variance:", response3)
            st.write("Final diff:", response4)
            # k=st.dataframe(response)
            # st.write("Raw table:", k)
            # with st.chat_message("assistant"):
            #     st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            #     response = agent.run('Filter  df1,df2 for the {Year} based on Period_Ending column, then Find the sum of Beginning_Balance in df2 and USD amount from df1  after joining them on Account Number only, then subtract the sum with Ending_Balance in df2 and show only account number where difference is not 0',st.session_state.messages, callbacks=[st_cb])
            #     st.session_state.messages.append({"role": "assistant", "content": response})
            #     st.write(response)

       
            # Convert LangChain's response to a DataFrame
            # try:
            #    df_result = pd.read_json(response, orient='records')
            #    st.write("Resulting DataFrame:")
            #    st.dataframe(df_result)
            # except Exception as e:
            #    st.write("Error:", e)

# Display the resulting DataFrame
       
            # Run the agent
    

    # Print the response (for debugging purposes)
        

    # Assuming response is a string, you might want to parse it accordingly
    # For example, if it's a JSON string, you can use pd.read_json





    # elif option == "Period Over Period Comparison":
    #     st.header("Period Over Period Comparison Analysis")

    #     # Get the list of available columns
    #     columns = get_column_list(je)
    #     selected_column = st.selectbox("Select Column for Comparison", columns,index=None)

    #     st.write(f"Selected Column: {selected_column}")

    # Additional analysis based on user selection can be added here

# Run the Streamlit app
if __name__ == "__main__":
    main()
