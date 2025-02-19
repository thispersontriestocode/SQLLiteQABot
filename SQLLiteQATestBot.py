#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType






db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")


# In[2]:


from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


# In[3]:




# Load Mistral from Ollama
llm = OllamaLLM(model="mistral")


# In[4]:





query_prompt_template = PromptTemplate(
    input_variables=["query", "dialect", "top_k"],
    template="""Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.


**Instructions:**
- You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
- **Always** use column names explicitly (e.g., `COUNT(EmployeeId) AS EmployeeCount` instead of `COUNT(*)`).
- Do **not** explain the query.
- Do **not** include any additional text.
- Do **not** describe database tables unless explicitly asked.
- Only return the final SQL query.
- Always limit results to **{top_k}** unless specified otherwise.
- Do **not** add anything to the end of the table name (e.g 'Employee' instead of 'Employees'

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.



**User Question:** {query}

**SQL Query Output:**
```sql
"""
)


# In[5]:


import re

def extract_sql_query(text):
    """Extract SQL query from the LLM response."""
    match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else text.strip()

def write_query(state):
    """Generate a SQL query using the refined prompt."""
    prompt = query_prompt_template.format(
        query=state["question"],
        dialect=db.dialect,
        top_k=5
    )

    result = llm.invoke(prompt)

    return {"query": extract_sql_query(result)}


# In[6]:


write_query({"question": "How many Employees are there?"})  


# In[7]:





def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


# In[8]:


execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})


# In[9]:


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response}


# In[10]:




graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()


# In[11]:


for step in graph.stream(
    {"question": "How many employee are there?"}, stream_mode="updates"
):
    print(step)


# In[12]:



memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = {"configurable": {"thread_id": "1"}}


# In[13]:


for step in graph.stream(
    {"question": "How many employees are there?"},
    config,
    stream_mode="updates",
):
    print(step)

try:
    user_approval = input("Do you want to go to execute query? (yes/no): ")
except Exception:
    user_approval = "no"

if user_approval.lower() == "yes":
    # If approved, continue the graph execution
    for step in graph.stream(None, config, stream_mode="updates"):
        print(step)
else:
    print("Operation cancelled by user.")


# ## Below is code for an agent instead of a chain
# 
# 

# In[15]:




toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

tools


# In[16]:


prompt_template = PromptTemplate(
    input_variables=["dialect", "top_k", "input"],
    template="""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

**Instructions:**
- You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
- **Always** use column names explicitly (e.g., `COUNT(EmployeeId) AS EmployeeCount` instead of `COUNT(*)`).
- Do **not** explain the query.
- Do **not** include any additional text.
- Do **not** describe database tables unless explicitly asked.
- Only return the final SQL query.
- Always limit results to **{top_k}** unless specified otherwise.
- Do **not** add anything to the end of the table name (e.g 'Employee' instead of 'Employees')



To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.
Question: {input}
"""
)


# In[17]:


user_question =  "Which country's customers spent the most?"

system_message = prompt_template.format(dialect="SQLite", top_k=5, input = user_question)


# In[18]:



# ... (your database connection and LLM initialization) ...

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Refine tool descriptions (VERY IMPORTANT for ZERO_SHOT_REACT_DESCRIPTION):
for tool in tools:
    if tool.name == "QuerySQLDatabaseTool":
        tool.description = """
        Use this tool to execute SQL queries against the database. 
        Input: A detailed and CORRECT SQL query.
        Output: The result from the database.
        IMPORTANT: Before using this tool, ALWAYS use the 'QuerySQLCheckerTool' to validate your query.
        If the query is incorrect, the tool will return an error message.
        If you encounter an issue like 'Unknown column...', use the 'InfoSQLDatabaseTool' to get the correct table fields.
        """
    elif tool.name == "InfoSQLDatabaseTool":
        tool.description = """
        Use this tool to get information about the database schema and sample rows.
        Input: A comma-separated list of table names.
        Output: The schema and sample rows for those tables.
        Use this tool to understand the database structure or to find the correct column names for your queries.
        Call the 'ListSQLDatabaseTool' first to know what tables are available.
        """
    elif tool.name == "ListSQLDatabaseTool":
        tool.description = """
        Use this tool to get a list of available tables in the database.
        Input: None.
        Output: A list of table names.
        Use this tool before using 'InfoSQLDatabaseTool' to make sure the tables exist.
        """
    elif tool.name == "QuerySQLCheckerTool":
        tool.description = """
        Use this tool to check if your SQL query is correct BEFORE executing it with 'QuerySQLDatabaseTool'.
        Input: The SQL query you want to check.
        Output: The original query if it is correct, or a corrected query if there were mistakes.
        ALWAYS use this tool first.
        """


agent = initialize_agent(
    tools,
    llm,  # Your Ollama LLM instance
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    prompt=system_message,
)

# Example usage:
#agent.invoke( "Which country's customers spent the most?")


# In[19]:


#agent.invoke("Describe the playlisttrack table")


# In[20]:


#agent.invoke("What are 10 track names and their artists?")


# In[ ]:



# Global cleanup (runs when the whole script exits)
def global_cleanup():
    print("Global cleanup on exit.")
    # ... any global cleanup code (e.g., closing files, database connections) ...

atexit.register(global_cleanup)  # Register the cleanup function

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("My SQLLite Chatbot")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.text_input("Enter your message:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        try:
            result = proper_agent.invoke({"input": user_input, "k": 3, "page_content": ""})
            response = result['output']
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:  # Local cleanup (runs after each interaction)
            print("Local cleanup (if needed).")
            # ... any local cleanup code ...

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()