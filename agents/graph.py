"""Сборка агентского графа LangGraph."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.config import Settings
from agents.nodes import make_generate_node, make_grade_node, make_retrieve_node
from agents.rag import RagClient
from agents.state import AgentState

__all__ = ["build_graph"]


def build_graph(
    rag: RagClient,
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Собрать и скомпилировать граф.

    Пока граф линейный:

        START -> retrieve -> grade -> generate -> END

    Узел ``rewrite`` и условный переход появятся отдельным шагом, когда
    включим цикл переформулировки. Компиляция с ``checkpointer`` сразу
    поддерживает память диалога по ``thread_id``.
    """
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", make_retrieve_node(rag, settings))
    graph.add_node("grade", make_grade_node(settings))
    graph.add_node("generate", make_generate_node(settings))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)
