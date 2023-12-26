# retrieve best checkpoint on valid data
model=`ls cv/wiki_lstm_700hidden_3layer/*.pt | python best_model.py`

cat data/wiki/test.lower.txt \
| python truecase.py \
    $model \
    -beamsize 10 \
    -verbose 0 \
    -gpuid 0 \
> data/wiki/output.txt

# calculate performance on test set
python word_eval.py data/wiki/test.txt cv/wiki_lstm_700hidden_3layer/output.txt