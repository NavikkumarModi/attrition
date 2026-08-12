PAPER = no-regret-is-not-no-harm

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f *.aux *.log *.out *.toc

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all clean distclean
