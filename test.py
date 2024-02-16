import streamlit as st
import json
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from datetime import datetime
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
        years = list(range(je['Period_Ending'].dt.year.min(), je['Period_Ending'].dt.year.max() + 1))
        selected_year = st.selectbox("Select Year Ending Date", years, index=None)

        st.write(f"Selected Year: {selected_year}")

        # LangChain - OpenAI setup
        from custom_pandas_agent import create_pandas_dataframe_agent_custom
        llmt = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        prompt_template_name = PromptTemplate(
            input_variables=['DataFrame1', 'DataFrame2', 'DataFrame3', 'Year'],
            template="Find the difference between the Ending Balance in TB and sum of Beginning Balance and USD amount for the {Year} based on Period_Ending after joining them on Account Number only. Generate a dataframe with account number, Beginning balance, USD Amount and Ending Balance where the difference generated earlier is not equal to 0. Show the resulting DataFrame."
        )
        agent = create_pandas_dataframe_agent_custom(llm=llmt, df=[je, tb, coa], prompt=prompt_template_name, verbose=True)

        executor = AgentExecutor(agent, llm=llmt, handle_parsing_errors=True)

        # Run the agent
        response = executor.run(input={'DataFrame1': je, 'DataFrame2': tb, 'DataFrame3': coa, 'Year': selected_year})
        st.write("Raw Response:", response)

        # Convert LangChain's response to a DataFrame
        try:
            df_result = pd.read_json(response, orient='records')
            st.write("Resulting DataFrame:")
            st.dataframe(df_result)
        except Exception as e:
            st.write("Error:", e)

if __name__ == "__main__":
    main()