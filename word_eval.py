import sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train a character-level language model')
parser.add_argument('-origin', required=True, help='path of file origin to be compared')
parser.add_argument('-pred', required=True, help='path of result file predicted')
parser.add_argument('-verbose', type=int, default=1, help='set to 0 to ONLY print the sampled text, no diagnostics')
parser.add_argument('-log', default='log_wrong_prediction.txt', help='sentences wrong predicted will be logged')

opt = parser.parse_args()
verbose = opt.verbose

def gprint(s):
    s = s.replace('\n', '<n>')
    if verbose:
        print(s, file=sys.stderr)

with open(opt.origin, encoding='utf-8') as gold, \
     open(opt.pred, encoding='utf-8') as pred:
    gold_sent = gold.readlines()
    pred_sent = pred.readlines()

f_log = open(opt.log, "w", encoding="utf8")
wrong_pred = []
num_correct = 0
num_changed_correct = 0
num_gold = 0
num_proposed = 0
total = 0
gprint("====================================================")
progress = range(len(pred_sent))
progress = tqdm(progress, desc="Evaluation")
#gprint("{Origin} : {Prediction}")
for i in progress:
    words = gold_sent[i].strip().split()
    pred_words = pred_sent[i].strip().split()
    for k in range(len(words)):
        if pred_words[k] != pred_words[k].lower():
            num_proposed += 1
        if words[k] != words[k].lower():
            num_gold += 1
        if words[k] == pred_words[k]:
            num_correct += 1
            if words[k] != words[k].lower():
                num_changed_correct += 1
        else:
            #gprint("{} : {}".format(words[k], pred_words[k]))
            wrong_pred.append(words[k] + " => " + pred_words[k] + "\n")
            wrong_pred.append(gold_sent[i].replace("\n", "") + "\n")
            wrong_pred.append(pred_sent[i].replace("\n", "") + "\n\n")

    total += len(words)
    gprint("------------------- Current Accuracy: %.2f%%" % (num_correct / total * 100.0))

gprint("====================================================")
print("Writting logs...")
f_log.write("".join(wrong_pred))
f_log.close()
acc = num_correct * 100.0 / total
try:
    P = float(num_changed_correct)/num_proposed
    R = float(num_changed_correct)/num_gold
    F = 2*P*R/(P+R)
except:
    P = 0
    R = 0
    F = 0
print("============ Result ============")
print('Accuracy: {:.2f}'.format(acc))
print('Precision: {:.2f}'.format(P*100))
print('Recall: {:.2f}'.format(R*100))
print('F1: {:.2f}'.format(F*100))
print("================================")
