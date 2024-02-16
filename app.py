# Import necessary libraries
from dotenv import find_dotenv, load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
# from langchain_experimental.llms import LangchainBase
# from langchain_openai import la
import streamlit as st
from datetime import datetime
# from custom_pandas_agent import create_pandas_dataframe_agent
# from custom_pandas_agent import CustomChatOpenAI
from custom_pandas_agent import create_pandas_dataframe_agent_custom

import pandas as pd
class LangchainChatOpenAI(OpenAI):

    def __init__(self, model="gpt-3.5-turbo-0613", temperature=0):
        super().__init__(model=model, temperature=temperature)

# Load environment variables
load_dotenv(find_dotenv())

# Read sample data
je = pd.read_excel(r"C:\Users\iamku\Documents\Sample\JE.xlsx")
tb = pd.read_excel(r"C:\Users\iamku\Documents\Sample\TB.xlsx")
coa = pd.read_excel(r"C:\Users\iamku\Documents\Sample\COA.xlsx")

# Set target column
target_column = 'posted date'
target_column_index = je.columns.get_loc(target_column)
je = je.iloc[:, :target_column_index + 1]

# Get date range
mindate = je['Period_Ending'].min()
maxdate = je['Period_Ending'].max()

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
        years = list(range(je['Period_Ending'].dt.year.min(), je['Period_Ending'].dt.year.max() + 1),)
        selected_year = st.selectbox("Select Year Ending Date", years, index=None)
  

        st.write(f"Selected Year: {selected_year}")
        # llmt = CustomChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        prompt_template_name = PromptTemplate(input_variables=['DataFrame1', 'DataFrame2', 'DataFrame3','Year'], template="Find the difference between the Ending Balance in TB and sum of Beginning Balance and USD amount for the {Year} based on Period_Ending after joining them on Account Number only. Generate a  dataframe with account number, Beginning balance, USD Amount and Ending Balance where the difference generated earlier is not equal to 0. Show the resulting DataFrame."
)
        # prompt_template_name = PromptTemplate(
        #     input_variables=['Year'],
        #     template="Find the difference between the Ending Balance in TB and sum of Beginning Balance and USD amount for the {Year} based on Period_Ending after joining them on Account Number. Generate a pandas dataframe with account number, Beginning balance, USD Amount and Ending Balance where the difference generated earlier is not equal to 0 "
        # )
        # agent = create_pandas_dataframe_agent(llm=llmt, df=[je, tb, coa], prompt=prompt_template_name, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
        # llm_dict = {'llm': llmt}
        agent = create_pandas_dataframe_agent_custom( df=[je, tb, coa],model="gpt-3.5-turbo-0613", temperature=0, prompt=prompt_template_name, verbose=True)
        executor = AgentExecutor(agent, handle_parsing_errors=True)
        response = executor.run(input={'DataFrame1': je, 'DataFrame2': tb, 'DataFrame3': coa,'Year':selected_year })
        st.write("Raw Response:", response)

       
            # Convert LangChain's response to a DataFrame
        try:
            df_result = pd.read_json(response, orient='records')
            st.write("Resulting DataFrame:")
            st.dataframe(df_result)
        except Exception as e:
            st.write("Error:", e)

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
