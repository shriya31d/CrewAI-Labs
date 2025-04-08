import os
from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field
from typing import List, Optional
from src.pydantic_models.SightseeingPlan import SightSeeingPlan
from src.pydantic_models.TravelItinerary import TravelItinerary
from src.tools.custom_tool import add_actual_dates_in_itinerary

@CrewBase
class TripPlannerCrew():
    """TripPlanner crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @agent
    def destination_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['destination_expert'],
            tools=[SerperDevTool(n_results=5), ScrapeWebsiteTool()], # Example of custom tool, loaded at the beginning of file
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def itinerary_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config['itinerary_compiler'],
            tools=[add_actual_dates_in_itinerary],
            verbose=True,
            allow_delegation=False,
        )

    #Add hotel_recommender agent here, use custom tool : get_recommended_hotels to get list of hotels

    @task
    def sightseeing_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['sightseeing_planning_task'],
            agent=self.destination_expert(),
            output_pydantic = SightSeeingPlan
        )
    
    #Add hotel_recommendation_task here 


    @task
    def itinerary_compilation_task(self) -> Task:
        return Task(
            config=self.tasks_config['itinerary_compilation_task'],
            agent=self.itinerary_compiler(),
            output_json=TravelItinerary
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TripPlanner crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            verbose = True,
            process=Process.sequential,
            # manager_agent= self.manager
        )