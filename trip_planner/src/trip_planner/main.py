#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from trip_planner.crew import TripPlannerCrew

import mlflow

mlflow.crewai.autolog()
# Optional: Set a tracking URI and an experiment name if you have a tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CrewAI")

def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'destination': 'Kashmir, India',
        'age': 31,
        'start_date': "13-04-2025",
        'trip_duration': '5 days'
    }
    result = TripPlannerCrew().crew().kickoff(inputs=inputs)
    print(result)