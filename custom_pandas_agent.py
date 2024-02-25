from langchain.agents.agent_types import AgentType
from langchain.agents.agent_types import create_pandas_dataframe_agent as create_pandas_dataframe_agent_original

def create_pandas_dataframe_agent_custom(df, model="gpt-3.5-turbo-0613", temperature=0, prompt=None, verbose=True):
    return create_pandas_dataframe_agent_original(llm=ChatOpenAI(model=model, temperature=temperature), df=df, prompt=prompt, verbose=verbose)