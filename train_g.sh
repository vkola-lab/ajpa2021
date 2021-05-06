export CUDA_VISIBLE_DEVICES=0
python src/main.py \
--n_class 4 \
--data_path "data/" \
--model_path "outputs/saved_models/" \
--log_path "outputs/runs/" \
--task_name "global_only" \
--mode 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \