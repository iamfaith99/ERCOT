# Research Manuscript Documentation

This directory contains the research paper manuscript and supporting documentation for the HayekNet battery bidding strategies project.

## 📄 Files

- **`manuscript.qmd`**: Main Quarto manuscript (primary research paper)
- **`references.bib`**: BibTeX bibliography file with citations
- **`Proposal.md`**: Original project proposal
- **`Research paper outline.md`**: Initial paper structure outline

## 🔨 Building the Manuscript

### Quick Start

From the project root directory:

```bash
make paper
```

This will generate:
- **`manuscript.pdf`**: Publication-ready PDF
- **`manuscript.html`**: Interactive HTML version with code folding

### Manual Build

If you prefer to build manually:

```bash
cd docs
quarto render manuscript.qmd --to pdf
quarto render manuscript.qmd --to html
```

### View Output

```bash
open docs/manuscript.pdf      # macOS
open docs/manuscript.html     # macOS
```

## 📊 Current Status

### ✅ Complete Sections

1. **Introduction** - Motivation, research questions, contributions
2. **Market Background** - ERCOT structure, RTC+B changes
3. **Data and Methods** - Data sources, battery model, trading strategies, metrics
4. **Results (Partial)** - Week 1 analysis with 5 days of data
5. **Discussion** - Economic implications, operational insights, limitations
6. **Conclusion** - Summary, implications, future work

### 🚧 In Progress (TODO Markers)

1. **Literature Review** - Needs expansion with more references
2. **Results - Forecasting Performance** - Awaiting ARIMA/LSTM implementation
3. **Results - RL Performance** - Awaiting Phase 1 training improvements
4. **Figures** - Need to generate key plots from data
5. **Appendices** - Mathematical formulations, data processing details

## 📝 Key Findings (Week 1)

### Economic Viability
- **Energy arbitrage alone**: -$11,583/day (including degradation)
- **RTC+B projected**: +$2,600 to +21,100/day (with ancillary services)
- **Conclusion**: AS participation is **essential** for economic viability

### Operational Insights
- **Optimal volatility**: 40-50% CoV (not maximum!)
- **SOC utilization**: 65-75% target range (r ≈ 0.85 correlation with profit)
- **Capture efficiency**: Only 3-22% of theoretical maximum

### RTC+B Implications
- Forecasting becomes mandatory (15-30 min ahead binding bids)
- Co-optimization unlocks 5-10x revenue increase
- Simple reactive strategies become obsolete

## 🔬 Reproducibility

The manuscript follows reproducibility best practices (per Rules #6, #7, #22, #23):

- All data sources documented
- Processing pipeline described
- Code available (TODO: add GitHub link)
- Configuration versioned
- Seeds specified for stochastic processes

### Environment

Paper requires:
- **Quarto** ≥ 1.7 (✅ installed: v1.7.29)
- **LaTeX** (for PDF rendering)
- **Python** (for executing code chunks if needed)

## 📚 Bibliography Management

### Adding References

Edit `references.bib` using BibTeX format:

```bibtex
@article{author2024paper,
  title={Paper Title},
  author={Author, Name},
  journal={Journal Name},
  year={2024},
  ...
}
```

### Citing in Manuscript

Use `@citekey` format:

```markdown
Previous work [@author2024paper] demonstrated...
Multiple citations [@author2024paper; @other2023work]...
```

## 🎯 Next Steps

### This Week (Oct 5-12)
1. ✅ Create manuscript structure
2. ⏳ Generate key figures (volatility-profitability, SOC-profitability)
3. ⏳ Implement Phase 1 improvements (training skip logic)
4. ⏳ Collect Week 2 data

### Next 2 Weeks (Oct 12-26)
1. ⏳ Expand literature review
2. ⏳ Implement forecast models (ARIMA/LSTM)
3. ⏳ Complete RL training improvements
4. ⏳ Add AS bidding simulation

### Month End (Oct 26-31)
1. ⏳ Generate all figures and tables
2. ⏳ Complete TODOs in manuscript
3. ⏳ First complete draft
4. ⏳ Internal review

## 📋 TODO Tracking

### Priority 1 (Critical for Draft)
- [ ] Generate volatility vs profitability scatter plot
- [ ] Generate SOC utilization vs profitability plot
- [ ] Add daily LMP distribution figures
- [ ] Add battery SOC trajectory plots
- [ ] Expand literature review (20+ references)

### Priority 2 (Important for Quality)
- [ ] Implement theoretical maximum calculator
- [ ] Add capture efficiency time series
- [ ] Create RTC+B revenue waterfall chart
- [ ] Expand comparison to ERCOT studies
- [ ] Add mathematical formulations appendix

### Priority 3 (Nice to Have)
- [ ] Add interactive HTML figures
- [ ] Create supplementary materials
- [ ] Add code snippets for key algorithms
- [ ] Create presentation slides from manuscript

## 🎨 Manuscript Style

Following academic conventions:
- **Abstract**: 200-250 words (✅ 224 words)
- **Length**: Target 8-12 pages for conference, 15-25 for journal
- **Sections**: Standard IMRAD structure
- **Citations**: Author-year format
- **Equations**: Numbered, referenced in text
- **Tables/Figures**: Numbered, captioned, referenced

## 💡 Tips

### Incremental Writing
The manuscript is designed as a "living document":
- Start with sections you can write now
- Leave `[TODO: ...]` markers for future content
- Update incrementally as more data arrives
- Re-render frequently to catch formatting issues

### Quarto Features
- Code chunks can be executed (set `echo: true` to show code)
- Figures auto-numbered and referenced with `@fig-label`
- Tables auto-numbered with `@tbl-label`
- Equations referenced with `@eq-label`
- Cross-references work in both PDF and HTML

### Preview While Writing
```bash
quarto preview manuscript.qmd  # Live reload in browser
```

## 📞 Support

For Quarto documentation: https://quarto.org/docs/

For HayekNet project questions: [TODO: Add contact]

---

**Last Updated**: October 5, 2025  
**Status**: Week 1 draft with preliminary findings  
**Next Review**: October 12, 2025 (after Week 2 data)
