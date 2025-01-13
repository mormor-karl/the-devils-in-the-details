# with the IB distinction, KB-BERT
python3 reannotate_iob.py ../../SWELL_Fixed/ ./data/k_only_iobs/ --k_fold 10 --model KB/bert-base-swedish-cased >  ./meta/k_only_iobs_meta.txt
python3 reannotate_iob.py ../../SWELL_Fixed/ ./data/k_detailed/ --k_fold 10 --model KB/bert-base-swedish-cased --detailed_iob >  ./meta/k_detailed_meta.txt
python3 reannotate_iob.py ../../SWELL_Fixed/ ./data/k_general_pseudo/ --k_fold 10 --model KB/bert-base-swedish-cased --detailed_iob --general_pseudo >  ./meta/k_general_pseudo_meta.txt

# without the IB distinction, KB-BERT
# only_iobs are the same regardless of whether --no_ib is called or not
python3 reannotate_iob.py ../../SWELL_Fixed/ ./data/k_detailed_no_ib/ --k_fold 10 --model KB/bert-base-swedish-cased --detailed_iob --no_ib >  ./meta/k_detailed_meta_no_ib.txt
python3 reannotate_iob.py ../../SWELL_Fixed/ ./data/k_general_pseudo_no_ib/ --k_fold 10 --model KB/bert-base-swedish-cased --detailed_iob --general_pseudo --no_ib >  ./meta/k_general_pseudo_meta_no_ib.txt