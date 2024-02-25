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
# os.chmod('C:/autoexec.bat', 0777)

def display_all_inputs(input_list):
    st.subheader("All Inputs:")
    all_inputs_text = "\n".join(input_list)
    st.text_area("Inputs:", all_inputs_text, height=200)

def path():
     img_path=os.path.join(os.getcwd(),r"exports\charts\temp_chart.png").replace("\\","/")
    #  img_path="E:/Projects/LedgerLLM/exports/charts/temp_chart.png"
     print(img_path)
     return img_path

path()


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
# img_path="E:/Projects/LedgerLLM/exports/charts/temp_chart.png"


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

            df = SmartDatalake( [je,tb,coa], config={ "llm": llm})
            response1 = df.chat(f'Filter  je,tb for the {selected_year} based on Period_Ending column and join them based of Account Num')
            response2 = response1.chat(f'Find the sum of Beginning_Balance and USD amount and create a new column Total_Beginning ')
            response3 = response2.chat(f'Subtract the Ending_Balance with Total_Beginning and create a new column Variance and round the variance')
            response4 = response3.chat(f'Replace all NaN values with 0 and Show account number,Beginning_Balance,USD_Amount,sum, Ending_Balance and Variance for only those rows where variance is not 0.0 and show the output dataset as a downloadable dataframe ')
            response5=response4.chat('Show Total no of rows as No of Variance Rows,sum of Variance column, sum of beginning balance column, sum of USD Amount column as JE activity and sum of Ending Balance column in a tabular format with column 1 as Values and column 2 as total and shows all totals as integers not as scientific numbers ')

            column_names = response4.columns
            response4 = pd.DataFrame(data=response4, columns=column_names)
            st.subheader("Stats and Final Reconciliation Data",divider='grey')
            st.write(response5)
            st.divider() 
            st.write(response4)

    elif option == "Smart Trends":
        
        img_path=path()
        if os.path.isfile(img_path):
                    os.remove(img_path) 
        st.caption("Period Over Period Comparison Analysis")
        columns = je.select_dtypes(include='float').columns.tolist()
        selected_column = st.selectbox("Select Column for Comparison", columns,index=None)
        st.divider() 
        if selected_column:
            
            tr=SmartDataframe(je,config={"llm": llm,"verbose":True,"open_charts":False})
            # if os.path.isfile(img_path):
            #         os.remove(img_path)
            col1, col2 = st.columns(2)
            test2=tr.chat(f"Show the Period Ending year and total sum of {selected_column} per year in a tabular format as integers not as scientific numbers and dont show the index column")
            test=tr.chat(f'Plot period over period graph between Period Ending year and total sum of {selected_column} per year and show data values and data labels unit and each bar in different color')

            st.subheader("Data and Plot",divider='grey')
            st.write( test2)
            st.divider()
            print(img_path)
            if os.path.exists(img_path):
                 print('going to print')
                 print(img_path)
                 img_path=os.path.normpath(img_path)
                 img=Image.open(img_path)
                 st.image(img, caption='Period Over Period Graph')
                 os.remove(img_path)
            else:
                 print("lol")
    elif option == "Smart Graph":
            img_path=path()
            st.caption("Smart Graph")  
            if os.path.exists(img_path):
                    os.remove(img_path)   
            
            all_inputs = []
            df = SmartDatalake( [je,tb], config={ "llm": llm,"response_parser": StreamlitResponse,"enable_cache": False,"verbose":True,"open_charts":False})
            a=st.text_input("Ask to plot data :")
            if a:
                if os.path.exists(img_path):
                    os.remove(img_path)
                b=df.chat(a)
                tt=df.last_code_generated
                c=df.chat(f'Using {tt} show the data generated in a tabular format')
                if os.path.exists(img_path):
                    img=Image.open(img_path)
                    st.image(img)
    elif option == "Smart Explorer":
            st.caption("Smart Explorer")    
            all_inputs = []
            df = SmartDatalake( [je,tb], config={ "llm": llm,"response_parser": StreamlitResponse,"enable_cache": False,"verbose":True})
            a=st.text_input("Explore your data")
            st.divider()
            if a:
                b=df.chat(a)
                st.write("Here is your data", b)
                st.divider()
                with st.expander("Code Generated"):
                    st.code(df.last_code_executed)                

# Run the Streamlit app
if __name__ == "__main__":
    main()
