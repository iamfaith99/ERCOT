# HayekNet Graduate Research Makefile

.PHONY: help activate daily quick run run-analysis analyze setup-automation jupyter paper clean clean-cache clean-runs

help:
	@echo "🎓 HayekNet Graduate Research Commands"
	@echo "======================================="
	@echo ""
	@echo "📅 Daily Research Workflow:"
	@echo "  daily         - Run full daily research workflow"
	@echo "  quick         - Quick test run (1 hour data)"
	@echo ""
	@echo "🚀 System Execution:"
	@echo "  run           - Run basic simulation"
	@echo "  run-analysis  - Run simulation with full analysis"
	@echo "  analyze       - Generate analysis for latest run"
	@echo ""
	@echo "🔧 Setup & Tools:"
	@echo "  activate      - Show activation command"
	@echo "  setup-automation - Configure daily 2 AM collection"
	@echo "  jupyter       - Start Jupyter Lab"
	@echo ""
	@echo "📝 Paper & Documentation:"
	@echo "  paper         - Generate research manuscript (PDF + HTML)"
	@echo ""
	@echo "🗑️  Cleanup:"
	@echo "  clean-cache   - Clear old cached data (>7 days)"
	@echo "  clean-runs    - Remove old simulation results"
	@echo "  clean         - Full cleanup (WARNING: removes environment)"

activate:
	@echo ""
	@echo "🐍 To activate HayekNet environment:"
	@echo "   source activate_hayeknet.sh"
	@echo ""

daily:
	@echo "📅 Running daily research workflow..."
	@echo ""
	@bash -c "export HAYEKNET_SKIP_CHECKS=1; export PYTHON_JULIACALL_HANDLE_SIGNALS=no; source activate_hayeknet.sh && python scripts/daily_research_workflow.py"
	@echo ""
	@echo "✅ Daily workflow complete!"
	@echo "📖 View journal: cat research/journal/journal_$$(date +%Y-%m-%d).md"

quick:
	@echo "⚡ Running quick test (1 hour data)..."
	@bash -c "export HAYEKNET_SKIP_CHECKS=1; export PYTHON_JULIACALL_HANDLE_SIGNALS=no; source activate_hayeknet.sh && python scripts/daily_research_workflow.py --quick"

run:
	@echo "🚀 Running HayekNet simulation..."
	@bash -c "export PYTHON_JULIACALL_HANDLE_SIGNALS=no; source activate_hayeknet.sh && python -m python.main"

run-analysis:
	@echo "🚀 Running HayekNet with full analysis..."
	@bash -c "export PYTHON_JULIACALL_HANDLE_SIGNALS=no; source activate_hayeknet.sh && python -m python.main_with_analysis"

analyze:
	@echo "📊 Generating analysis for latest run..."
	@bash -c "export PYTHON_JULIACALL_HANDLE_SIGNALS=no; source activate_hayeknet.sh && python scripts/analyze_latest.py"

setup-automation:
	@echo "🔧 Setting up daily 2 AM automation..."
	@bash -c "./scripts/setup_daily_collection.sh"

jupyter:
	@echo "📊 Starting Jupyter Lab..."
	@bash -c "export PYTHON_JULIACALL_HANDLE_SIGNALS=no; source activate_hayeknet.sh && jupyter lab --no-browser --port=8888"

paper:
	@echo "📝 Generating research manuscript..."
	@bash -c "cd docs && quarto render manuscript.qmd --to pdf"
	@bash -c "cd docs && quarto render manuscript.qmd --to html"
	@echo ""
	@echo "✅ Manuscript generated!"
	@echo "📄 PDF: docs/manuscript.pdf"
	@echo "🌐 HTML: docs/manuscript.html"
	@echo ""
	@echo "To view:"
	@echo "  open docs/manuscript.pdf"
	@echo "  open docs/manuscript.html"

clean-cache:
	@echo "🗑️  Clearing old cached data (>7 days)..."
	@bash -c "export PYTHON_JULIACALL_HANDLE_SIGNALS=no; source activate_hayeknet.sh && python -c 'from python.data import ERCOTDataClient; ERCOTDataClient().clear_cache(older_than_days=7)'"
	@echo "✅ Cache cleaned"

clean-runs:
	@echo "🗑️  Removing old simulation results..."
	@rm -rf runs/*
	@echo "✅ Results removed"

clean:
	@echo "⚠️  WARNING: This will remove the conda environment!"
	@echo "🗑️  Removing conda environment..."
	@conda env remove -n hayeknet -y || true
	@echo "✅ Environment removed"
