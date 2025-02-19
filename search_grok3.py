import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Grok3SearchAgent:
    def __init__(self):
        # Initialize the LLM (OpenAI in this case)
        self.llm = OpenAI(temperature=0.5)
        
        # Configure SerpAPI wrapper
        self.search = SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
            params={
                "engine": "google",
                "google_domain": "google.com",
                "gl": "us",
                "hl": "en"
            }
        )
        
        # Create custom prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["query", "results"],
            template="""
            User query: {query}
            
            Search results: {results}
            
            Please provide a concise summary of the relevant information about Grok 3 based on the search results. Focus on key details and recent developments. Limit the response to 150 words or less.
            """
        )
        
        # Create summary chain
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        
        # Define tools
        self.tools = [
            Tool(
                name="Search",
                func=self.search.run,
                description="Useful for searching current information about Grok 3 on the web."
            )
        ]
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Pull the ReAct prompt template from LangChain Hub
        prompt = hub.pull("hwchase17/react")
        
        # Create the ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def search_and_summarize(self, query: str) -> str:
        """
        Perform a search for Grok 3 information and return a summarized response.
        """
        try:
            # Run the agent to get search results
            search_results = self.agent_executor.invoke({
                "input": f"Search for current information about Grok 3: {query}"
            })
            
            # Extract the search results from the agent's output
            results = search_results.get('output', '')
            
            # Generate summary using the summary chain
            summary = self.summary_chain.run(
                query=query,
                results=results
            )
            
            return summary.strip()
            
        except Exception as e:
            return f"Error processing request: {str(e)}"

def main():
    # Initialize the agent
    agent = Grok3SearchAgent()
    
    print("Welcome to the Grok 3 Information Search Agent!")
    print("Enter your query about Grok 3 (or 'quit' to exit)")
    
    while True:
        query = input("\nQuery: ").strip()
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
            
        if not query:
            print("Please enter a valid query.")
            continue
            
        print("\nSearching...")
        result = agent.search_and_summarize(query)
        print("\nResult:")
        print(result)

if __name__ == "__main__":
    main()