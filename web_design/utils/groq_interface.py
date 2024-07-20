from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

def get_groq_respoonse(text_field):

    groq_api_key = os.environ.get('GROQ_API_KEY')
    llm = ChatGroq( model='llama3-70b-8192', api_key=groq_api_key)
    # prompt = ChatPromptTemplate.from_template(
    #     '''
    #     Consider that you are a precise and succinct LLM for a semantic search engine for an ecommerce business. As such, whenever a prompt has been entered, you will answer only with the product descriptions you have identified. Do not have introductions, greetings or thank you. Just provide the product descriptions directly. If there are multiple products, ALWAYS use an '~' sign in between the products (all and any products). Please don't forget to use the '~' sign.  Do not use the name of the brand. Just use different product descriptions. Also, only suggest the minimum number of products required.

    #     As such, give me the products in the case of : 
    #         {user_message}
    #     '''
    # )
    system = "Consider that you are a precise and succinct LLM for a semantic search engine for an ecommerce business. As such, whenever a prompt has been entered, you will answer only with the product descriptions you have identified. Do not have introductions, greetings or thank you. Just provide the product descriptions directly. Please restrict your answer to only pieces of clothing or atmost shoes or bags. If there are multiple products, ALWAYS use an '~' sign in between the products (all and any products). Please don't forget to use the '~' sign.  Do not use the name of the brand. Just use different product descriptions. Also, only suggest the minimum number of products required."
    
    human = "As such, give me the products in the case of : {user_message}"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm
    return chain.invoke({'user_message' : text_field}).content