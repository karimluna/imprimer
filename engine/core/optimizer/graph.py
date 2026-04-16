"""
LangGraph optimization graph —> the outer control loop.
This is the top-level entry point for prompt optimization.

Graph structure:
  generator -> evaluator -> controller -> (generator | END)

The inner loop (Optuna TPE) lives inside the generator node.
The outer loop (reachability threshold) is managed by this graph.
Together they implement a two-level control hierarchy:
  - Inner: find the best mutation in this cycle (Optuna)
  - Outer: decide if the result is good enough to stop (LangGraph)
"""

from langgraph import StateGraph, END

from core.optimizer.state import PromptState