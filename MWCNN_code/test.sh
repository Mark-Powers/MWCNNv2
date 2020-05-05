
python main.py --model MWCNN --scale 15 --n_feats 64 --save_results --print_model --n_colors 3 --test_only --self_ensemble --resume -1 --pre_train experiment/MWCNN_HDR_window_1/model/ --data_test HDRTest --task_type clip --ev 0.5

python3 stitch.py
