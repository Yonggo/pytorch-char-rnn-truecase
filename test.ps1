$model_dir = "cv/deu_news_2018_1M_lstm_700hidden_3layer_lr0.0005"
$eval_dir_name = "wiki_300K"
$log_path = "data/Dataset/eval/$eval_dir_name/log.txt"
$to_be_predicted = "data/Dataset/eval/$eval_dir_name/eval.lower.txt"
$predicted_result_path = "data/Dataset/eval/$eval_dir_name/output.txt"
$validator_path = "data/Dataset/eval/$eval_dir_name/eval.txt"

$model=(ls $model_dir/*.pt -name | python best_model.py)
$model = "$model_dir/$model"

python truecase_line_by_line.py `
    $model `
    -beamsize 10 `
    -verbose 1 `
    -gpuid 0 `
    -target $to_be_predicted `
    -result $predicted_result_path `

python word_eval.py -origin $validator_path -pred $predicted_result_path -verbos 0 -log $log_path