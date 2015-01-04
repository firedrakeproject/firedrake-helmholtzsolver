TEX=pdflatex
LATEXMK=latexmk
MAIN=GravityWaves
SOURCES=$(MAIN).tex Makefile $(shell ls *.tex) $(shell ls *.bib)

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SOURCES)
	$(LATEXMK) -pdf $(MAIN).tex

.PHONY: clean
clean:
	$(LATEXMK) -C $(MAIN).tex
