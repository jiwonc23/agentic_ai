from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_search_agent():
    # Initialize the search tool
    search = SerpAPIWrapper()

    # Initialize the language model
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") 

    # Create a list of tools for the agent
    search_tool = Tool(        
        name = "Search",
        func = search.run,
        description = "Useful for searching information on the internet"
    )

    # Initialize the agent
    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )   

    return agent

def process_query(query: str):
    try:
        agent = create_search_agent()
        response = agent.run(f"Search for information about: {query}. Provide a concise summary.")
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # Example usage
    query = input("Enter your search query: ")
    result = process_query(query)
    print("\nResult:", result)