from g2pM import G2pM

SPLIT_TOKEN = "▁"

sent_file = "./data/test.sent"
label_file = "./data/test.lb"

sents = open(sent_file, "r") 
labels = open(label_file, "r")
model = G2pM()

for sent, label in zip(sents, labels):
    poly_idx = sent.index(SPLIT_TOKEN)
    # replace special token _
    sent = sent.replace(SPLIT_TOKEN, "").strip()
    poly_char = sent[poly_idx]
    label = label.strip()
    prons = model(sent)
    print("target pinyin of {}: {}".format(poly_char, label))
    print("sentence :{}".format(sent))
    print("pinyins: {}".format(prons))
    print("Please press enter button for next prediction")
    _ = input()
