# necessary imports
import sqlite3
import streamlit as st
import os
import openai
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, WebsiteSearchTool
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black
import functools
#rectifying the error


class AdvancedMarketResearchApp:
    def __init__(self, openai_api_key=None, serper_api_key=None):
        # Initialize session state variables if not present
        if 'results' not in st.session_state:
            st.session_state.results = None

        # Set API keys if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            openai.api_key = openai_api_key

        if serper_api_key:
            os.environ["SERPER_API_KEY"] = serper_api_key

        # Initialize web search tools
        try:
            self.serper_tool = SerperDevTool()
            self.websearch_tool = WebsiteSearchTool()
        except Exception as e:
            st.error(f"Error initializing search tools: {e}")
            self.serper_tool = None
            self.websearch_tool = None

    def create_agents(self, company, industry):
        # Check if tools are initialized
        if not self.serper_tool or not self.websearch_tool:
            raise ValueError("Search tools not properly initialized. Check API keys.")

        # Industry Research Agent
        industry_researcher = Agent(
            role='Advanced Industry Analyst',
            goal=f'Conduct comprehensive research on {industry} and {company}, identifying market trends, competitive landscape, and strategic opportunities',
            backstory=f'A seasoned industry analyst with deep expertise in {industry}, specializing in technological innovation and market dynamics.',
            tools=[self.serper_tool, self.websearch_tool],
            verbose=False,
            allow_delegation=True
        )

        # Use Case Generation Agent
        use_case_agent = Agent(
            role='AI Strategy Consultant',
            goal='Generate innovative, data-driven AI/ML use cases with clear business value and implementation strategies',
            backstory='A strategic technology consultant who transforms complex business challenges into actionable AI solutions, drawing from extensive cross-industry experience.',
            tools=[self.serper_tool, self.websearch_tool],
            verbose=False,
            allow_delegation=True
        )

        # Resource Collection Agent
        resource_agent = Agent(
            role='AI Resource Curator',
            goal='Identify and compile cutting-edge datasets, research papers, and implementation resources for proposed AI solutions',
            backstory='A meticulous technology researcher who discovers and validates the most relevant and recent technological resources across global platforms.',
            tools=[self.serper_tool, self.websearch_tool],
            verbose=False,
            allow_delegation=True
        )

        return industry_researcher, use_case_agent, resource_agent

    @functools.lru_cache(maxsize=100)
    def run_research(self, company, industry):
        try:
            # Create agents
            agents = self.create_agents(company, industry)

            # Create tasks
            tasks = self.create_tasks(agents, company, industry)

            # Create and run crew
            crew = Crew(
                agents=list(agents),
                tasks=tasks,
                verbose=False,
                max_rpm=8
            )

            # Kickoff research
            result = crew.kickoff()

            return result
        except Exception as e:
            st.error(f"An error occurred during research: {e}")
            return None

    def create_tasks(self, agents, company, industry):
        industry_researcher, use_case_agent, resource_agent = agents

        # Research Task
        research_task = Task(
            description=f'Conduct comprehensive analysis of {company} in the {industry} sector',
            agent=industry_researcher,
            expected_output='Detailed industry and company analysis report'
        )

        # Use Case Generation Task
        use_case_task = Task(
            description=f'Generate AI/ML use cases for {company} in {industry}',
            agent=use_case_agent,
            expected_output='Innovative AI/ML use cases with implementation strategies'
        )

        # Resource Collection Task
        resource_task = Task(
            description='Compile relevant resources and datasets',
            agent=resource_agent,
            expected_output='Curated list of resources and implementation guides'
        )

        return [research_task, use_case_task, resource_task]

    def generate_pdf_report(self, company, industry, result):
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=18)

            styles = getSampleStyleSheet()
            # Create custom styles
            title_style = styles['Title']
            heading1_style = ParagraphStyle(
                'Heading1',
                parent=styles['Heading1'],
                fontSize=14,
                textColor=black
            )
            heading2_style = ParagraphStyle(
                'Heading2',
                parent=styles['Heading2'],
                fontSize=12,
                textColor=black
            )
            normal_style = styles['BodyText']

            story = []
            # Adding the title
            story.append(Paragraph(f"AI Use Case Research: {company} - {industry}", title_style))
            story.append(Spacer(1, 12))

            # If result is from CrewAI tasks output
            if hasattr(result, 'tasks_output') and result.tasks_output:
                for task_output in result.tasks_output:
                    raw_output = getattr(task_output, 'raw', str(task_output))

                    if raw_output:
                        sections = raw_output.split('\n\n')

                        for section in sections:
                            if not section.strip():
                                continue

                            if section.startswith('####'):
                                section_text = section.strip('#').strip()
                                story.append(Paragraph(section_text, heading2_style))
                            elif section.startswith('###'):
                                section_text = section.strip('#').strip()
                                story.append(Paragraph(section_text, heading1_style))
                            else:
                                story.append(Paragraph(section, normal_style))

                            story.append(Spacer(1, 6))

            # Fallback if no tasks output
            elif result:
                story.append(Paragraph(str(result), normal_style))
            else:
                story.append(Paragraph("No research results available.", normal_style))

            # Build the PDF
            doc.build(story)

            pdf_buffer = buffer.getvalue()
            buffer.close()

            return pdf_buffer

        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            return None

def main():
    # Setting the page configuration
    st.set_page_config(
        page_title="AI Use Case Generator",
        page_icon="🤖",
        layout="wide"
    )

    # Title and description
    st.title("StratAI")
    st.markdown("### 🚀 AI Use Case Research Generator")

    # Input columns
    col1, col2 = st.columns(2)

    with col1:
        company = st.text_input("Company Name", placeholder="e.g., Tata")

    with col2:
        industry = st.text_input("Industry", placeholder="e.g., Automotive")

    # API Key Inputs in Sidebar
    st.sidebar.title("API Configuration")

    # Serper Dev API Key Input
    serper_api_key = st.sidebar.text_input("Enter Serper Dev API Key", type="password")

    # OpenAI API Key Input
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    # Research button
    if st.button("Generate AI Use Cases", type="primary"):
        if company and industry and serper_api_key and openai_api_key:
            try:
                with st.spinner('Conducting Market Research...'):
                    app = AdvancedMarketResearchApp(
                        openai_api_key=openai_api_key,
                        serper_api_key=serper_api_key
                    )
                    result = app.run_research(company, industry)
                    st.session_state.results = result
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter Company, Industry, Serper Dev API Key, and OpenAI API Key")

    # Display results if available
    if 'results' in st.session_state and st.session_state.results:
        st.header("🔍 Research Findings")

        result = st.session_state.results

        if hasattr(result, 'tasks_output') and result.tasks_output:
            for task_output in result.tasks_output:
                raw_output = getattr(task_output, 'raw', str(task_output))

                if raw_output:
                    sections = raw_output.split('\n\n')

                    for section in sections:
                        if not section.strip():
                            continue

                        if section.startswith('####'):
                            st.subheader(section.strip('#').strip())
                        elif section.startswith('###'):
                            st.markdown(f"### {section.strip('#').strip()}")
                        else:
                            st.write(section.strip())

        else:
            st.info("No detailed output available yet.")

        if result:
            app = AdvancedMarketResearchApp(
                openai_api_key=openai_api_key,
                serper_api_key=serper_api_key
            )
            pdf_buffer = app.generate_pdf_report(company, industry, result)

            if pdf_buffer:
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_buffer,
                    file_name=f"{company}_{industry}_AI_Use_Cases.pdf",
                    mime="application/pdf",
                    type="primary"
                )

    st.sidebar.info("""
    This AI-powered tool helps businesses:
    - Understand industry trends
    - Discover AI transformation opportunities
    - Find relevant implementation resources
    - Discover, Analyze, and Innovate with AI

    Powered by CrewAI and OpenAI
    """)

if __name__ == "__main__":
    main()
