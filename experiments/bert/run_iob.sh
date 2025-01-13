d_flag=''  # data directory
l_flag=''  # labels file
m_flag=''  # model name
o_flag=''  # output dir
e_flag=''  # num epochs
b_flag=''  # batch size

files=''
verbose='false'

print_usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options (all required):"
 echo " -d  Input directory"
 echo " -l  Labels file"
 echo " -m  Model name"
 echo " -o  Output directory"
 echo " -e  Number of epochs"
 echo " -b  Batch size"
 echo "task_type, max_sequence_length, save_steps, and seed are hard-coded"
 echo "only do_train and do_predict are enabled"
}

while getopts 'd:l:m:o:e:b:' flag; do
  case "${flag}" in
    d) d_flag=${OPTARG} ;;
    l) l_flag=${OPTARG} ;;
    m) m_flag=${OPTARG} ;;
    o) o_flag=${OPTARG} ;;
    e) e_flag=${OPTARG} ;;
    b) b_flag=${OPTARG} ;;
    *) print_usage
       exit 1 ;;
  esac
done

# export BERT_MODEL=KB/bert-base-swedish-cased
# export OUTPUT_DIR=testytesty
# export BATCH_SIZE=8
# export NUM_EPOCHS=3

python3 run_ner.py \
--task_type NER \
--data_dir $d_flag \
--labels $l_flag \
--model_name_or_path $m_flag \
--output_dir $o_flag \
--max_seq_length  512 \
--num_train_epochs $e_flag \
--per_device_train_batch_size $b_flag \
--save_steps 750 \
--seed 1 \
--do_train \
--do_predict
