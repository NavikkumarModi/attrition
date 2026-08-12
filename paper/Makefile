PAPER = no-regret-is-not-no-harm

all: figures $(PAPER).pdf

figures:
	cd .. && PYTHONPATH=. python3 paper/make_figures.py

$(PAPER).pdf: $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f *.aux *.log *.out *.toc

distclean: clean
	rm -f $(PAPER).pdf fig*.pdf

.PHONY: all figures clean distclean
