PAPER  = no-regret-is-not-no-harm
ARXIV  = arxiv-no-regret-is-not-no-harm
JMLR   = jmlr-consumable-action-sets

# All three builds share body.tex, so they cannot drift apart. The JMLR version
# additionally pulls in sections_long.tex via the \LONGFORM macro.
all: figures $(PAPER).pdf $(ARXIV).pdf $(JMLR).pdf

figures:
	cd .. && PYTHONPATH=. python3 paper/make_figures.py

%.pdf: %.tex body.tex sections_long.tex
	pdflatex -interaction=nonstopmode $<
	pdflatex -interaction=nonstopmode $<

# arXiv upload bundle. No .bbl needed: the bibliography is inline.
arxiv-submission.tar.gz: $(ARXIV).tex arxiv.sty body.tex sections_long.tex
	tar czf $@ $(ARXIV).tex arxiv.sty body.tex sections_long.tex fig*.pdf

# JMLR submission bundle. Swap the preamble for \documentclass[wcp]{jmlr} and
# add jmlr.cls (distributed from jmlr.org) before submitting.
jmlr-submission.tar.gz: $(JMLR).tex body.tex sections_long.tex
	tar czf $@ $(JMLR).tex body.tex sections_long.tex fig*.pdf

clean:
	rm -f *.aux *.log *.out *.toc

distclean: clean
	rm -f $(PAPER).pdf $(ARXIV).pdf $(JMLR).pdf fig*.pdf *.tar.gz

.PHONY: all figures clean distclean
