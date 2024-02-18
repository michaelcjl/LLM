from bert import *
from data_prepare import *

# batch_size = 6
# batch_data = make_data(token_list, n_data=batch_size)
# batch_tensor = [torch.LongTensor(ele) for ele in zip(*batch_data)]

# dataset = BERTDataset(*batch_tensor)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model = BERT(n_layers)
# lr = 1e-3
# epochs = 500
# criterion = nn.CrossEntropyLoss()
# optimizer = Adadelta(model.parameters(), lr=lr)
# model.to(device)


# training
def train():
    for epoch in range(epochs):
        for one_batch in dataloader:
            input_ids, segment_ids, masked_tokens, masked_pos, is_next = [ele.to(device) for ele in one_batch]

            logits_cls, logits_lm = model(input_ids, segment_ids, masked_pos)
            loss_cls = criterion(logits_cls, is_next)
            loss_lm = criterion(logits_lm.view(-1, max_vocab), masked_tokens.view(-1))
            loss_lm = (loss_lm.float()).mean()
            loss = loss_cls + loss_lm
            if (epoch + 1) % 10 == 0:
                print(f'Epoch:{epoch + 1} \t loss: {loss:.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



# Using one sentence to test
def eval():
    test_data_idx = 3
    model.eval()
    with torch.no_grad():
        input_ids, segment_ids, masked_tokens, masked_pos, is_next = batch_data[test_data_idx]
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).to(device)
        masked_pos = torch.LongTensor(masked_pos).unsqueeze(0).to(device)
        masked_tokens = torch.LongTensor(masked_tokens).unsqueeze(0).to(device)
        logits_cls, logits_lm = model(input_ids, segment_ids, masked_pos)
        input_ids, segment_ids, masked_tokens, masked_pos, is_next = batch_data[test_data_idx]
        print("========================================================")
        print("Masked data:")
        masked_sentence = [idx2word[w] for w in input_ids if idx2word[w] != '[PAD]']
        print(masked_sentence)

        # logits_lm: [batch, max_pred, max_vocab]
        # logits_cls: [batch, 2]
        cpu = torch.device('cpu')
        pred_mask = logits_lm.data.max(2)[1][0].to(cpu).numpy()
        pred_next = logits_cls.data.max(1)[1].data.to(cpu).numpy()[0]

        bert_sentence = masked_sentence.copy()
        original_sentence = masked_sentence.copy()

        for i in range(len(masked_pos)):
            pos = masked_pos[i]
            if pos == 0:
                break
            print("i: {} , pos: {}, pred_mask[i]: {}", i, pos, pred_mask[i])
            bert_sentence[pos] = idx2word[pred_mask[i]]
            original_sentence[pos] = idx2word[masked_tokens[i]]

        print("BERT reconstructed:")
        print(bert_sentence)
        print("Original sentence:")
        print(original_sentence)

        print("===============Next Sentence Prediction===============")
        print(f'Two sentences are continuous? {True if is_next else False}')
        print(f'BERT predict: {True if pred_next else False}')




def train_new():
    model = BERT()
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 只计算mask位置的损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(100):
        optimizer.zero_grad()
        # logits_lm 语言词表的输出
        # logits_clsf 二分类的输出
        # logits_lm：[batch_size, max_pred, n_vocab]
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)## logits_lm 【6，5，29】 bs*max_pred*voca  logits_clsf:[6*2]
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens) # for masked LM ;masked_tokens [6,5]
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
    



if __name__ == "__main__":
    # train()
    # eval()
    train_new()
    # pass 

