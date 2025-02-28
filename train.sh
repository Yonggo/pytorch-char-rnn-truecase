python train.py \
  -data_dir data/wiki \
  -rnn_size 700 \
  -num_layers 3 \
  -dropout 0.25 \
  -batch_size 100 \
  -seq_length 50 \
  -max_epochs 30 \
  -learning_rate 0.001 \
  -checkpoint_dir cv/wiki_lstm_700hidden_3layer \
  -gpuid 0