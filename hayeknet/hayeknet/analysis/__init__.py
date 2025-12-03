"""Analysis and research components."""

from hayeknet.analysis.research import (
    AutomatedResearchAnalyst,
    ResearchInsight,
    run_automated_analysis,
)
from hayeknet.analysis.observations import ResearchObservationTracker
from hayeknet.analysis.qa import answer_and_save_questions, ResearchQuestionAnswerer, ResearchAnswer
from hayeknet.analysis.metrics import ResultsAnalyzer, compare_runs
from hayeknet.analysis.results import (
    ResultsReader,
    ResultsWriter,
    SimulationMetadata,
    SimulationResults,
)

__all__ = [
    "AutomatedResearchAnalyst",
    "ResearchInsight",
    "run_automated_analysis",
    "ResearchObservationTracker",
    "answer_and_save_questions",
    "ResearchQuestionAnswerer",
    "ResearchAnswer",
    "ResultsAnalyzer",
    "compare_runs",
    "ResultsReader",
    "ResultsWriter",
    "SimulationMetadata",
    "SimulationResults",
]

