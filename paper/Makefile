PAPER = no-regret-is-not-no-harm

# One consolidated document: body.tex is the main paper, sections_long.tex is
# pulled in via \LONGFORM as a clearly-marked appendix (see the "Appendices"
# divider before \appendix in $(PAPER).tex). This targets JMLR and arXiv with
# the same source, so there is nothing to keep in sync across builds.
all: figures $(PAPER).pdf

figures:
	cd .. && PYTHONPATH=. python3 paper/make_figures.py

$(PAPER).pdf: $(PAPER).tex body.tex sections_long.tex
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

# Submission bundle. For JMLR, swap the preamble for \documentclass[wcp]{jmlr}
# and add jmlr.cls (distributed from jmlr.org) before submitting -- body.tex
# and sections_long.tex need no changes either way.
submission.tar.gz: $(PAPER).tex body.tex sections_long.tex
	tar czf $@ $(PAPER).tex body.tex sections_long.tex fig*.pdf

clean:
	rm -f *.aux *.log *.out *.toc

distclean: clean
	rm -f $(PAPER).pdf fig*.pdf *.tar.gz

.PHONY: all figures clean distclean
