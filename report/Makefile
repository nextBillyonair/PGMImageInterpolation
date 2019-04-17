PDFLATEX:=pdflatex
BIBTEX:=bibtex


TEX_SRC=report.tex

${TEX_SRC:.tex=.pdf}:%.pdf: %.tex %.bib
	$(PDFLATEX) $<
	$(BIBTEX) ${<:.tex=}
	$(PDFLATEX) $<
	$(PDFLATEX) $<
	rm -f *.out *.aux *.log *.bbl *.blg

clean:
	rm -f *.out *.aux *.log *.bbl *.blg


.PHONY: clean
