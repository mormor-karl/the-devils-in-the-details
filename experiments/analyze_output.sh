python3 analyze_output.py ./results/nodalida/k_detailed/ model --bert_path ./bert/k_detailed/ --data_path ./data/k_detailed/ --k_fold
python3 analyze_output.py ./results/nodalida/k_detailed_no_ib/ model --bert_path ./bert/k_detailed_no_ib/ --data_path ./data/k_detailed_no_ib/ --k_fold

python3 analyze_output.py ./results/nodalida/k_general_pseudo/ model --bert_path ./bert/k_general_pseudo/ --data_path ./data/k_general_pseudo/ --k_fold
python3 analyze_output.py ./results/nodalida/k_general_pseudo_no_ib/ model --bert_path ./bert/k_general_pseudo_no_ib/ --data_path ./data/k_general_pseudo_no_ib/ --k_fold

python3 analyze_output.py ./results/nodalida/k_only_iobs/ model --bert_path ./bert/k_only_iobs/ --data_path ./data/k_only_iobs/ --k_fold
python3 analyze_output.py ./results/nodalida/k_only_iobs_no_ib/ model --bert_path ./bert/k_only_iobs_no_ib/ --data_path ./data/k_only_iobs_no_ib/ --k_fold


python3 analyze_output.py ./results/nodalida/merged/k_detailed/ model --bert_path ./bert/k_detailed/ --data_path ./data/k_detailed/ --k_fold --ignore_classes
python3 analyze_output.py ./results/nodalida/merged/k_detailed_no_ib/ model --bert_path ./bert/k_detailed_no_ib/ --data_path ./data/k_detailed_no_ib/ --k_fold --ignore_classes

python3 analyze_output.py ./results/nodalida/merged/k_general_pseudo/ model --bert_path ./bert/k_general_pseudo/ --data_path ./data/k_general_pseudo/ --k_fold --ignore_classes
python3 analyze_output.py ./results/nodalida/merged/k_general_pseudo_no_ib/ model --bert_path ./bert/k_general_pseudo_no_ib/ --data_path ./data/k_general_pseudo_no_ib/ --k_fold --ignore_classes

python3 analyze_output.py ./results/nodalida/merged/k_only_iobs/ model --bert_path ./bert/k_only_iobs/ --data_path ./data/k_only_iobs/ --k_fold --ignore_classes
python3 analyze_output.py ./results/nodalida/merged/k_only_iobs_no_ib/ model --bert_path ./bert/k_only_iobs_no_ib/ --data_path ./data/k_only_iobs_no_ib/ --k_fold --ignore_classes
