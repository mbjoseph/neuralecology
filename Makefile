# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


all: paper/neural_ecology.pdf

figs = fig/centroid-displacement.jpg \
	fig/occupancy_scatter.jpg \
	fig/persist-dist-plot.jpg \
	fig/roc-test.jpg \
	fig/route_tsne.jpg
	
clean_data = data/cleaned/bbs_counts.csv \
	data/cleaned/bbs_species.csv \
	data/cleaned/bbs_routes.csv 

agg_data = data/bbs_aggregated/bird.csv \
	data/bbs_aggregated/route.csv \
	data/bbs_aggregated/species.csv
	
model_comps = out/nll-comps.csv out/train-valid-nll.csv

paper/neural_ecology.pdf: $(figs) paper/neural_ecology.Rmd paper/library.bib \
	paper/title.sty paper/doc-prefix.tex paper/ecology-letters.csl \
	data/cleaned/bbs-summary.csv data/cleaned/clean_routes.csv \
	out/nll-comps.csv out/coverage_df.csv out/auc_df.csv out/dec_df.csv \
	out/cosine_sim.csv data/cleaned/bbs_species.csv out/select_paths.csv
		Rscript -e "rmarkdown::render('paper/neural_ecology.Rmd')"
	# replace paths in tex output (sed magic)
	sed "s?`pwd`/fig/??" paper/neural_ecology.tex > paper/neural_ecology_submit.tex


data/cleaned/bbs-summary.csv data/cleaned/bbs.csv data/cleaned/clean_routes.csv: R/04-clean-data.R $(clean_data) data/cleaned/routes.csv
		Rscript --vanilla R/04-clean-data.R

data/cleaned/routes.csv data/cleaned/routes.shp: R/03-extract-route-features.R data/NA_CEC_Eco_Level3.shp data/cleaned/bbs_routes.csv
		Rscript --vanilla R/03-extract-route-features.R

data/NA_CEC_Eco_Level3.shp: 
		wget ftp://newftp.epa.gov/EPADataCommons/ORD/Ecoregions/cec_na/NA_CEC_Eco_Level3.zip
		unzip -o -d data NA_CEC_Eco_Level3.zip
		rm NA_CEC_Eco_Level3.zip

$(clean_data): R/02-eda.R $(agg_data)
		Rscript --vanilla R/02-eda.R

$(agg_data): R/01-get-bbs-data.R
		Rscript --vanilla R/01-get-bbs-data.R

$(model_comps) out/route_embeddings.csv: R/06-compare-performance.R \
	R/05-single-species-models.R stan/dynamic-occupancy.stan \
	bbs-occupancy-model.py \
	utils.py dataset.py data/cleaned/bbs.csv data/cleaned/clean_routes.csv R/utils.R
		($(CONDA_ACTIVATE) neural-ecology ; python bbs-occupancy-model.py )
		Rscript --vanilla R/05-single-species-models.R
		Rscript --vanilla R/06-compare-performance.R

fig/roc-test.jpg out/auc_df.csv out/coverage_df.csv: R/utils.R \
	data/cleaned/clean_routes.csv data/cleaned/routes.shp \
	data/NA_CEC_Eco_Level3.shp $(model_comps)
		Rscript --vanilla R/07-test-set-checks.R

fig/occupancy_scatter.jpg fig/route_tsne.jpg out/cosine_sim.csv: out/route_embeddings.csv \
	data/cleaned/clean_routes.csv R/08-plot-embeddings.R
		Rscript --vanilla R/08-plot-embeddings.R

out/z_mles.csv out/z_finite_sample.csv: R/09-viterbi.R out/coverage_df.csv \
	out/route_embeddings.csv
		Rscript --vanilla R/09-viterbi.R

fig/centroid-displacement.jpg out/dec_df.csv fig/persist-dist-plot.jpg out/select_paths.csv: R/10-analyze-states.R \
	out/z_mles.csv out/z_finite_sample.csv data/cleaned/clean_routes.csv \
	data/cleaned/bbs_species.csv data/cleaned/routes.shp data/NA_CEC_Eco_Level3.shp
		Rscript --vanilla R/10-analyze-states.R

clean: 
		rm -f paper/neural_ecology.pdf
		rm -f $(figs)
		rm -f $(clean_data)
		rm -f $(agg_data)
		rm -f $(model_comps)
		rm -rf out
		mkdir out
		touch out/.gitignore 
		rm -rf data
		rm -f stan/*.rds
		rm -f data/cleaned/bbs-summary.csv data/cleaned/bbs.csv data/cleaned/clean_routes.csv
		rm -f data/cleaned/routes.csv data/cleaned/routes.shp
		rm -f data/NA_CEC_Eco_Level3.shp
		rm -f out/route_embeddings.csv
		rm -f out/auc_df.csv out/coverage_df.csv
		rm -f out/z_mles.csv out/z_finite_sample.csv
		rm -f out/dec_df.csv
		rm -f Rplots.pdf
		
