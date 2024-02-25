# Import necessary libraries
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from datetime import datetime
load_dotenv(find_dotenv())
import pandas as pd
from pandasai import SmartDataframe
from pandasai.prompts import AbstractPrompt
from pandasai import SmartDatalake
import pandas as pd
from pandasai.llm import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.responses.response_parser import ResponseParser
from PIL import Image
from langchain_core.output_parsers import BaseOutputParser
import os


def display_all_inputs(input_list):
    st.subheader("All Inputs:")
    all_inputs_text = "\n".join(input_list)
    st.text_area("Inputs:", all_inputs_text, height=200)

# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent


class MyCustomPrompt(AbstractPrompt):
    @property
    def template(self):
        return """Filter  df1,df2 for the {x} based on Period_Ending column , then Find the sum of Beginning_Balance in df2 and USD amount from df1  after joining them on Account Number only, then subtract the sum with Ending_Balance in df2 and show only account number,Beginning_Balance,USD_Amount, Ending_Balance and difference in a tabular format where difference is not 0 for the selected {x}"""

    # def setup(self, kwargs):

    #     # This method is called before the prompt is initialized
    #     # You can use it to setup your prompt and pass any additional
    #     # variables to the template
    #     self.set_var("x", kwargs["x"])

class PandasDataFrame(ResponseParser):

    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        # Returns Pandas Dataframe instead of SmartDataFrame
        return result["value"]


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


llm = OpenAI(temperature=0)
# Streamlit app
def main():
    st.header("LedgerSmart",divider='grey')

    # Sidebar with options
    option = st.sidebar.selectbox("Select Analysis Type", ["AutoReconcile", "Smart Trends","Smart Explorer","Smart Graph"], index=None)

    # Main content based on user selection
    if option == "AutoReconcile":

        st.caption("GL Recon Analysis")
        

        # Get the last three years
        current_year = datetime.now().year
        years = list(range(je['Period_Ending'].dt.year.min(), je['Period_Ending'].dt.year.max() + 1),)
        # st.caption("Select Year Ending Date")
        selected_year = st.selectbox("Select Year Ending Date", years, index=None)
        st.divider() 
  

        
        if selected_year:
            # st.write(f"Selected Year: {selected_year}")
            # df = SmartDatalake( [df1,df2,df3], config={ "llm": llm,   "custom_prompts": { "generate_python_code": MyCustomPrompt(x=selected_year)},"response_parser": PandasDataFrame})
            # df = SmartDatalake( [df1,df2,df3], config={ "llm": llm,  "response_parser": PandasDataFrame})
            df = SmartDatalake( [je,tb,coa], config={ "llm": llm})
            response1 = df.chat(f'Filter  je,tb for the {selected_year} based on Period_Ending column and join them based of Account Num')
            response2 = response1.chat(f'Find the sum of Beginning_Balance and USD amount and create a new column Total_Beginning ')
            response3 = response2.chat(f'Subtract the Ending_Balance with Total_Beginning and create a new column Variance and round the variance')
            response4 = response3.chat(f'Replace all NaN values with 0 and Show account number,Beginning_Balance,USD_Amount,sum, Ending_Balance and Variance for only those rows where variance is not 0.0 and show the output dataset as a downloadable dataframe ')
            # response4= pd.DataFrame(response4)
            # response4 = SmartDataframe( [response4], config={ "llm": llm})

            response5=response4.chat('Show Total no of rows as No of Variance Rows,sum of Variance column, sum of beginning balance column, sum of USD Amount column as JE activity and sum of Ending Balance column in a tabular format with column 1 as Values and column 2 as total and shows all totals as integers not as scientific numbers ')
            # repsonse=df.chat(f"Filter  df1,df2 for the {selected_year} based on Period_Ending column , then Find the integer sum of Beginning_Balance in df2 and USD amount from df1  after joining them on Account Number only, then integer subtract the sum with Ending_Balance in df2 as integers and show only account number,Beginning_Balance,USD_Amount,sum, Ending_Balance and difference in a tabular format where difference is not 0 for the selected {selected_year}")

            
            # t=SmartDataframe(je,config={"llm": llm,"response_parser": StreamlitResponse})
            # "save_charts":True,"save_charts_path":'E:/Projects/LedgerLLM/'
            # "save_charts_path":'E:\Projects\LedgerLLM\'
            # test=t.chat('Plot period over period graph between Period Ending year and total no of records per year and show data values ')
            # test=test.impute_missing_values()
            # # test=test.plot_correlation_heatmap()
            # # dft = SmartDataframe(df1)
            # test=test.plot_correlation_heatmap()
            # st.write("Final Dataframe", repsonse)
            # st.write("Filter:", response1)
            # st.write("Total Beginning:", response2)
            # st.write("Variance:", response3)
            column_names = response4.columns
            response4 = pd.DataFrame(data=response4, columns=column_names)
            st.subheader("Stats and Final Reconciliation Data",divider='grey')

            st.write(response5)
            st.divider() 
            # st.write("How:", response5)
            
            st.write(response4)
            # st.write("Final image:", test)
            # img=Image.open("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
            # st.image(img, caption='Sunrise by the mountains')
    elif option == "Smart Trends":
        st.caption("Period Over Period Comparison Analysis")
        columns = je.select_dtypes(include='float').columns.tolist()
        selected_column = st.selectbox("Select Column for Comparison", columns,index=None)
        st.divider() 
        # st.write(f"Selected Column: {selected_column}")
        if selected_column:
            print('here')
            tr=SmartDataframe(je,config={"llm": llm,"verbose":True})
            col1, col2 = st.columns(2)
            # r1=tr.chat(f'based on the values in {selected_column} Find the data type of {selected_column} from df1 ')

            # st.write("Column Type:", r1)
            # test=t.chat('Plot period over period graph between Period Ending year and total no of records per year and show data values ')
            test2=tr.chat(f"Show the Period Ending year and total sum of {selected_column} per year in a tabular format as integers not as scientific numbers and dont show the index column")
            test=tr.chat(f'Plot period over period graph between Period Ending year and total sum of {selected_column} per year and show data values and data labels unit and each bar in different color')
            # st.write("Final image:", test)
            st.subheader("Data and Plot",divider='grey')
            st.write( test2)
            st.divider()
            img=Image.open("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
            st.image(img, caption='Period Over Period Graph')
            # col1.header("Data")
            # col1.write(test2,use_column_width=True)
            # col2.header(f"Period Over Period {selected_column}")
            # col2.image(img,use_column_width=True)

            os.remove("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
    elif option == "Smart Graph":
            st.caption("Smart Graph")     
                 
            all_inputs = []
            df = SmartDatalake( [je,tb], config={ "llm": llm,"response_parser": StreamlitResponse,"enable_cache": False,"verbose":True})
            a=st.text_input("Ask to plot data :")
            
            
            if a:
                
                 

                if os.path.exists("E:/Projects/LedgerLLM/exports/charts/temp_chart.png"):
                    os.remove("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
                b=df.chat(a)
                tt=df.last_code_generated
                c=df.chat(f'Using {tt} show the data generated in a tabular format')
                # st.write("Data Generated", c)
                
                if os.path.exists("E:/Projects/LedgerLLM/exports/charts/temp_chart.png"):
                    img=Image.open("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
                    st.image(img)
            
                    # os.remove("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
                # st.write("response", b)
                # with st.expander("Code Generated"):
                #     st.code(df.last_code_executed)
    elif option == "Smart Explorer":
            st.caption("Smart Explorer")    
                   
            all_inputs = []
            df = SmartDatalake( [je,tb], config={ "llm": llm,"response_parser": StreamlitResponse,"enable_cache": False,"verbose":True})
            a=st.text_input("Explore your data")
            st.divider()
            
            if a:

                
                 

                # if os.path.exists("E:/Projects/LedgerLLM/exports/charts/temp_chart.png"):
                #     os.remove("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
                b=df.chat(a)
                # tt=df.last_code_generated
                # c=df.chat(f'Using {tt} show the data generated in a tabular format')
                # st.write("Data Generated", c)
                
                # if os.path.exists("E:/Projects/LedgerLLM/exports/charts/temp_chart.png"):
                #     img=Image.open("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
                #     st.image(img)
            
                #     # os.remove("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
                st.write("Here is your data", b)
                st.divider()
                with st.expander("Code Generated"):
                    st.code(df.last_code_executed)                
            # os.remove("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")

# if os.path.exists("E:/Projects/LedgerLLM/exports/charts/temp_chart.png"):
  
#   os.remove("E:/Projects/LedgerLLM/exports/charts/temp_chart.png")
# else:
#   print("The file does not exist")
            # try:
            #    df_result = pd.read_json(response4, orient='records')
            #    st.write("Resulting DataFrame:")
            #    st.dataframe(df_result)
            # except Exception as e:
            #    st.write("Error:", e)
           
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

# class CustomAnalysisPrompt(Prompt):
#     text = """
# You are provided with a dataset that contains sales data by brand across various regions. Here's the metadata for the given pandas DataFrames:

# {dataframes}

# Given this data, please follow these steps:
# 0. Acknowledge the user's query and provide context for the analysis.
# 1. **Data Analysis**: < custom instructions > 
# 2. **Opportunity Identification**: < custom instructions > 
# 3. **Reasoning**: < custom instructions > 
# 4. **Recommendations**: < custom instructions > 
# 5. **Output**: Return a dictionary with:
#    - type (possible values: "text", "number", "dataframe", "plot")
#    - value (can be a string, a dataframe, or the path of the plot, NOT a dictionary)
#    Example: {{ "type": "text", "value": < custom instructions > }}
# ``python
# def analyze_data(dfs: list[pd.DataFrame]) -> dict:
#    # Code goes here (do not add comments)

# # Declare a result variable
# result = analyze_data(dfs)
# ``

# Using the provided dataframes (`dfs`), update the Python code based on the user's query:
# {conversation}

# # Updated code:
# # """
