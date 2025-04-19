.PHONY: summary venv mamba
SHELL := /bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
MPIPRECMD := $(shell command -v mpirun >/dev/null 2>&1 && echo "mpirun -n 10")
PROJNAME := bspinn
MAMBABASE := ${PROJBASE}/mamba
PREFENV := venv

##########################################################
###########    Summarizing the HDF Results   #############
##########################################################
summary: summary.lazy

summary.lazy:
	source ./activate ${PREFENV} && python bspinn/summary.py --lazy

summary.full:
	source ./activate ${PREFENV} && python bspinn/summary.py

##########################################################
#####################      Venv     ######################
##########################################################
venv:
	python -m venv venv
	source ./activate venv && python -m pip install --upgrade pip
	source ./activate venv && python -m pip install -r requirements.txt
	source ./activate venv && python -m pip install -e .
	rm -rf *.egg-info

##########################################################
#####################      Mamba     #####################
##########################################################
mamba:
	mkdir -p ${MAMBABASE}
	cd ${MAMBABASE}; \
	curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba;
	set -e; \
	export MAMBA_ROOT_PREFIX=${MAMBABASE}; \
	eval "$$(${MAMBABASE}/bin/micromamba shell hook -s posix)"; \
	micromamba activate; \
	micromamba create  -y -n ${PROJNAME} python=3.11 -c conda-forge; \
	micromamba activate ${PROJNAME}; \
	micromamba install -y -c conda-forge openssh; \
	export TMPDIR=${PROJBASE}/pip; mkdir -p $${TMPDIR}; \
	python -m pip install --upgrade pip; \
	python -m pip install jupyter; \
	python -m pip install -r requirements.txt; \
	python -m pip install -e .; \
	rm -rf *.egg-info; \
	rm -r $${TMPDIR};

##########################################################
####################      Cleanup     ####################
##########################################################
fix_crlf:
	find ${PROJBASE} -maxdepth 3 -type f -name "*.md5" \
	  -o -name "*.py" -o -name "*.sh" -o -name "*.json" | xargs dos2unix

clean:
	@MYPROJBASE=${PROJBASE}; \
	RESBASE=$$MYPROJBASE"/results"; \
	CSVTREE=""; \
	cd $$MYPROJBASE; \
	while [[ -d $$RESBASE/$$CSVTREE ]] ; do \
		cd $$RESBASE/$$CSVTREE; \
		if [[ n$$CSVTREE == "n" ]] ; then \
		  MYSEP=""; \
		else \
		 MYSEP="/"; \
		fi; \
		CSVTREE="$$CSVTREE""$$MYSEP"$$(ls -Art | tail -n 1); \
	done; \
	cd $$MYPROJBASE; \
	CSVTREE="$${CSVTREE%_part*}" ; \
	CSVTREE="$${CSVTREE%.csv*}" ; \
	while [[ n"$$CSVTREE" != "n" ]] ; do \
	  if ls ./results/$$CSVTREE* 1> /dev/null 2>&1; then \
		  :; \
		else \
		  echo "File patterns " ./results/$$CSVTREE* "do not exist."; \
			break; \
		fi; \
		echo rm -rf; \
		for f in ./results/$$CSVTREE* ; do \
		  echo "  " $$f; \
		done ;\
		for f in ./storage/$$CSVTREE* ; do \
		  echo "  " $$f; \
		done ;\
		read -p "?[y/n] " PERM; \
		if [[ $$PERM == "y" ]] ; then \
			echo rm -rf ./results/$$CSVTREE*; \
			rm -rf ./results/$$CSVTREE*; \
			CSVTREE=""; \
		elif [[ $$PERM == "n" ]] ; then \
		  echo "----------"; \
		  echo "Enter the config tree to remove [Leave Empty to Exit]: "; \
			echo "  Ex: 00_scratch/scratch1 "; \
			echo "  Ex: 01_firth_1layer/firth_1layer "; \
			read -p "? " CSVTREE; \
		else \
		  CSVTREE=""; \
		fi; \
	done
