$lr=0.0005

python train.py `
  -data_dir data/Dataset/deu_news_2018_1M-sentences `
  -checkpoint_dir cv/deu_news_2018_1M_lstm_700hidden_3layer_lr$lr `
  -rnn_size 700 `
  -num_layers 3 `
  -dropout 0.25 `
  -batch_size 100 `
  -seq_length 50 `
  -max_epochs 30 `
  -learning_rate $lr `
  -gpuid 0