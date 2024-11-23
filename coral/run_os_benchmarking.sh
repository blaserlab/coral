# qsub -q "gpu.q" -pe smp 1 -l gpu_mem=40G -l h_rt=336:00:00 -cwd -j yes -o logs/os_benchmarking.log run_os_benchmarking.sh

source /workspace/python/anaconda3/etc/profile.d/conda.sh
conda activate coral
# module load cuda
export PYTHONPATH=../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

echo "GPU allocation: $CUDA_VISIBLE_DEVICES"

gemma_model="/workspace/refdata/LLM_curation_data/models/models--google--gemma-7b-it/snapshots/9c5798d27f588501ce1e108079d2a19e4c3a2353/"
annot_data_dir='/workspace/refdata/LLM_curation_data/coral_data/annotated/'
fdata='breastca_unannotated.csv'
fout='breastca_out.csv'
dir_data='/workspace/refdata/LLM_curation_data/coral_data/unannotated/data'
dir_out='../coral_output/'

echo "Creating inference data"
python -u dataprocessor/create_inference_data.py \
-annot_data_dir $annot_data_dir \
-fdata $fdata \
-dir_data $dir_data

echo "Running inference. Current model: ${gemma_model}"
python -u benchmarking/open_source_benchmarking.py \
-fdata $fdata \
-fout $fout \
-dir_data $dir_data \
-dir_out $dir_out \
-model_name_or_path "$gemma_model" \
-batch_size 2

echo "Evaluating"
python -u benchmarking/evaluate_model.py -fdata $fdata -fout $fout -dir_data $dir_data -dir_out $dir_out
