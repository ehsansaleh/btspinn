# all: summary figures tables
# figures: plot_loss_vs_dim
# prep: venv dl_lite

.PHONY: summary venv figures tables
SHELL := /bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
MPIPRECMD := $(shell command -v mpirun >/dev/null 2>&1 && echo "mpirun -n 10")

##########################################################
###########      Summarizing CSV Results     #############
##########################################################

summary: summary.lazy

summary.lazy:
	source .env.sh && python bspinn/summary.py --lazy

summary.full:
	source .env.sh && python bspinn/summary.py

##########################################################
####################  Figures/Tables #####################
##########################################################

# plot_loss_vs_dim:
# 	source .env.sh && python plotters/plot_loss_vs_dim.py
# tables:
# 	source .env.sh && python utils/summ2tbls.py

##########################################################
####################      VENV     #######################
##########################################################

venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip
	source venv/bin/activate && python -m pip install torch==1.7.1+cu101 \
		torchvision==0.8.2+cu101 \
		-f https://download.pytorch.org/whl/torch_stable.html
	source venv/bin/activate && python -m pip install -r requirements.txt
	source venv/bin/activate && python -m pip install -e .
	rm -rf *.egg-info

##########################################################
####################    Downloads   ######################
##########################################################

# dl_lite:
# 	./results/download.sh

rsync:
	rsync -av --update --progress results joblogs ehsans2@cc-xfer.campuscluster.illinois.edu:/projects/west/ehsan/sciml/code_bspinn

##########################################################
####################      Clean     ######################
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