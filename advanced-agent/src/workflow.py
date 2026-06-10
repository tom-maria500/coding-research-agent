from typing import Dict, Any 
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, SystemMessage 
from .models import ResearchState, CompanyInfo, CompanyAnalysis
from .firecrawl import FireCrawlService
from .prompts import DeveloperToolsPrompts


class Workflow:
    def __init__(self):
        self.firecrawl = FireCrawlService()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.prompts = DeveloperToolsPrompts()
        self.worflow = self._build_workflow()

    def _build_workflow(self):
        # run state through the graph
        graph = StateGraph(ResearchState)
        # naming the nodes 
        graph.add_node("extract_tools", self._extract_tools_step)
        graph.add_node("research", self._research_step)
        graph.add_node("analyze", self._analyze_step)
        graph.set_entry_point("extract_tools")
        # set ordering with edges 
        graph.add_edge("extract_tools", "research")
        graph.add_edge("research", "analyze")
        # need end node
        graph.add_edge("analyze", END)
        return graph.compile()

    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        print(f"Finding articles about: {state.query}")

        article_query = f"{state.query} tools comparison best alternatives"
        search_results = self.firecrawl.search_companies(article_query, num_results=3)

        all_content = ""

        for result in search_results.web:
            # search results return objects, so get the url using getattr
            url = getattr(result, "url", "")

            if not url and getattr(result, "metadata", None):
                url = result.metadata.get("sourceURL", "") or result.metadata.get("url", "")

            if not url:
                continue

            scraped = self.firecrawl.scrape_company_page(url)

            if scraped:
                markdown = getattr(scraped, "markdown", "")

                if markdown:
                    # limit content added to model
                    all_content += markdown[:1500] + "\n\n"
        
        messages = [
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(state.query, all_content))
        ]

        try:
            response = self.llm.invoke(messages)
            tools_names = [
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip()
            ]
            print(f"Extracted tools: {', '.join(tools_names[:5])}")
            # will match to state model and update
            return {"extracted_tools": tools_names}
        except Exception as e:
            print(e)
            return {"extracted_tools": []}
        
    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis:
        structured_llm = self.llm.with_structured_output(CompanyAnalysis)

        messages = [
            SystemMessage(content=self.prompts.TOOL_ANALYSIS_SYSTEM),
            HumanMessage(content=self.prompts.tool_analysis_user(company_name, content))
        ]

        try:
            # return content as company analysis object 
            analysis = structured_llm.invoke(messages)
            return analysis
        except Exception as e:
            print(e)
            return CompanyAnalysis(
                pricing_model="Unknown",
                is_open_source=None,
                tech_stack=[],
                description="Failed",
                api_available=None,
                language_support=[],
                integration_capabilities=[],
            )
        
    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        extract_tools = getattr(state, "extracted_tools", [])

        if not extract_tools:
            print("No extracted tools found, falling back to direct search")
            search_results = self.firecrawl.search_companies(state.query, num_results=4)

            tool_names = [
                getattr(result, "title", None)
                or (
                    result.metadata.get("title", "Unknown")
                    if getattr(result, "metadata", None)
                    else "Unknown"
                )
                for result in search_results.web
            ]
        else:
            tool_names = extract_tools[:4]
        
        print(f"Researching specific tools {', '.join(tool_names)}")

        companies = []

        for tool_name in tool_names:
            tool_search_results = self.firecrawl.search_companies(
                tool_name + " official site",
                num_results=1
            )

            if tool_search_results and tool_search_results.web:
                result = tool_search_results.web[0]

                # search for url and get the content 
                url = getattr(result, "url", "")

                if not url and getattr(result, "metadata", None):
                    url = result.metadata.get("sourceURL", "") or result.metadata.get("url", "")

                company = CompanyInfo(
                    name=tool_name,
                    description="",
                    website=url,
                    tech_stack=[],
                    competitors=[]
                )

                scraped = self.firecrawl.scrape_company_page(url)

                if scraped:
                    content = getattr(scraped, "markdown", "")

                    if content:
                        analysis = self._analyze_company_content(company.name, content)

                        company.pricing_model = analysis.pricing_model
                        company.is_open_source = analysis.is_open_source
                        company.tech_stack = analysis.tech_stack
                        company.description = analysis.description
                        company.api_available = analysis.api_available
                        company.language_support = analysis.language_support
                        company.integration_capabilities = analysis.integration_capabilities

                companies.append(company)

        return {"companies": companies}
    
    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        print("Generating recommendations")

        companies_data = ", ".join([
            company.json() for company in state.companies
        ])

        messages = [
            SystemMessage(content=self.prompts.RECOMMENDATIONS_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(state.query, companies_data))
        ]

        try:
            response = self.llm.invoke(messages)
            return {"analysis": response.content}
        except Exception as e:
            print(e)
            return {"analysis": "Failed to generate analysis"}
        
    def run(self, query: str) -> ResearchState:
        initial_state = ResearchState(query=query)
        final_state = self.worflow.invoke(initial_state)
        # unpack dict into the object 
        return ResearchState(**final_state)