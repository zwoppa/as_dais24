# Databricks notebook source
# MAGIC %pip install databricks-genai-inference

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

# MAGIC %pip install langchain-community

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

dbutils.widgets.text("Support question", "Can you cancel my order?")
question = dbutils.widgets.get("Support question")

vsc = VectorSearchClient(disable_notice=True).get_index("test", "workspace.default.support")
dvs = DatabricksVectorSearch(vsc)

rel_docs = dvs.similarity_search(question)

context = [doc.page_content for doc in rel_docs]
full_context = "\n".join(context)

system_message = "You are a helpful service and support assistant that is helping with handling people's order requests. Typically people need help with cancelling orders, changing orders etc. When people have a request please include 'I will do that for you' and the action that you're helping with. You can assume the user has already provided necessary information for the relevant order. Do not say what steps to take, just fix them yourself and let the person do no interaction. Keep the response short."

messages = [
    SystemMessage(content=system_message),
    HumanMessage(content=f"{question} - Here is some relevant information: {full_context}"),
]
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=500, temperature=0.15)
response = chat_model.invoke(messages)

final_response = response.to_json()["kwargs"]["content"]

print(final_response)


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

dbutils.widgets.text("Support question", "Can you cancel my order?")
question = dbutils.widgets.get("Support question")

def chatbot_answer():
    vsc = VectorSearchClient(disable_notice=True).get_index("test", "workspace.default.support")
    dvs = DatabricksVectorSearch(vsc)

    rel_docs = dvs.similarity_search(question)

    context = [doc.page_content for doc in rel_docs]
    full_context = "\n".join(context)

    system_message = "You are a helpful service and support assistant that is helping with handling people's order requests. Typically people need help with cancelling orders, changing orders etc. When people have a request please include 'I will do that for you' and the action that you're helping with. You can assume the user has already provided necessary information for the relevant order. Do not say what steps to take, just fix them yourself and let the person do no interaction. Keep the response short."

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"{question} - Here is some relevant information: {full_context}"),
    ]
    chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=500, temperature=0.15)
    response = chat_model.invoke(messages)

    final_response = response.to_json()["kwargs"]["content"]
    return final_response



# COMMAND ----------

from IPython.display import Image 
from IPython.core.display import HTML 
Image(url= "https://github.com/zwoppa/as_dais24/blob/main/serviceflow.jpg?raw=true")

# COMMAND ----------


displayHTML(f"<h1>ServiceFlow</h1>")
displayHTML("<img src='https://github.com/zwoppa/as_dais24/blob/main/serviceflow.jpg?raw=true' style='width:600px;height:400px;'")
displayHTML(f"<h3>Question: {question}</h3>")
displayHTML(f"<h3>Answer: {chatbot_answer()}</h3>")
