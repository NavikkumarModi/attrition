PAPER  = no-regret-is-not-no-harm
ARXIV  = arxiv-no-regret-is-not-no-harm

all: figures $(PAPER).pdf $(ARXIV).pdf

figures:
	cd .. && PYTHONPATH=. python3 paper/make_figures.py

$(PAPER).pdf: $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

# arXiv submission version: uses arxiv.sty (Meshgi/Georgiev style, based on the
# NeurIPS layout). arXiv itself imposes no template; this is the community
# convention for preprints.
$(ARXIV).pdf: $(ARXIV).tex arxiv.sty
	pdflatex -interaction=nonstopmode $(ARXIV).tex
	pdflatex -interaction=nonstopmode $(ARXIV).tex

# Bundle for arXiv upload: source, style, figures. No .bbl needed (bibliography
# is inline via thebibliography).
arxiv-submission.tar.gz: $(ARXIV).tex arxiv.sty
	tar czf $@ $(ARXIV).tex arxiv.sty fig*.pdf

clean:
	rm -f *.aux *.log *.out *.toc

distclean: clean
	rm -f $(PAPER).pdf $(ARXIV).pdf fig*.pdf arxiv-submission.tar.gz

.PHONY: all figures clean distclean
