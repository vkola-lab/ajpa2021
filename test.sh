export CUDA_VISIBLE_DEVICES=0
python src/main.py \
--n_class 4 \
--data_path "data/" \
--eval_file "data/OSUWMC_test.txt" \
--model_path "outputs/saved_models/" \
--log_path "outputs/runs/" \
--task_name "testing" \
--mode 2 \
--batch_size 1 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \
--path_g "global_only.pth" \
--path_g2l "global2local.pth" \
--evaluation