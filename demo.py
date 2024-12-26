import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import jieba
from collections import Counter
import pickle
from tqdm import tqdm
from config import get_args
import warnings
import random

warnings.filterwarnings("ignore")

# 特殊标记
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

# 构建词汇表
def build_vocab(sentences, is_target=False, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        if is_target:
            tokens = jieba.lcut(sentence)
        else:
            tokens = sentence.split()
        counter.update(tokens)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# 数据预处理
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        # 源语言: 英文，使用空格分词
        src_tokens = [self.src_vocab.get(word, self.src_vocab[UNK_TOKEN]) for word in src_sentence.split()]
        
        # 目标语言: 中文，使用 jieba 分词，并添加 <sos> 和 <eos>
        tgt_tokens = [self.tgt_vocab[SOS_TOKEN]] + \
                     [self.tgt_vocab.get(word, self.tgt_vocab[UNK_TOKEN]) for word in jieba.lcut(tgt_sentence)] + \
                     [self.tgt_vocab[EOS_TOKEN]]

        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

# 自定义 collate_fn 以填充序列
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lengths = [len(seq) for seq in src_batch]
    tgt_lengths = [len(seq) for seq in tgt_batch]
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)

    padded_src = [torch.cat([seq, torch.zeros(max_src_len - len(seq)).long()]) for seq in src_batch]
    padded_tgt = [torch.cat([seq, torch.zeros(max_tgt_len - len(seq)).long()]) for seq in tgt_batch]

    return torch.stack(padded_src), torch.stack(padded_tgt)

# 模型定义
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [1, batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, emb_dim]

        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)  # [batch_size, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hid_dim]

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + hid_dim]
        output, hidden = self.rnn(rnn_input, hidden)  # output: [batch_size, 1, hid_dim]

        output = output.squeeze(1)  # [batch_size, hid_dim]
        context = context.squeeze(1)  # [batch_size, hid_dim]
        embedded = embedded.squeeze(1)  # [batch_size, emb_dim]

        output = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch_size, output_dim]
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt):
        encoder_outputs, hidden = self.encoder(src)
        input = tgt[:,0]  # <sos>
        outputs = torch.zeros(tgt.shape[0], tgt.shape[1], self.decoder.output_dim).to(self.device)

        for t in range(1, tgt.shape[1]):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:,t] = output
            input = tgt[:,t]

        return outputs

# 训练模型
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            # output: [batch_size, tgt_len, output_dim]
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)  # 排除 <sos>
            tgt = tgt[:,1:].reshape(-1)  # 排除 <sos>
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")

# 保存模型和词汇表
def save_model(model, src_vocab, tgt_vocab, path='model.pth', vocab_path='vocab.pkl'):
    torch.save(model.state_dict(), path)
    with open(vocab_path, 'wb') as f:
        pickle.dump({'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, f)

# 加载模型和词汇表
def load_model(path, vocab_path, INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, device):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    src_vocab = vocab['src_vocab']
    tgt_vocab = vocab['tgt_vocab']
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
    attention = Attention(HID_DIM)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, src_vocab, tgt_vocab

# 翻译函数
def translate(sentence, model, src_vocab, tgt_vocab, device, max_len=100):
    model.eval()
    with torch.no_grad():
        src_tokens = [src_vocab.get(word, src_vocab[UNK_TOKEN]) for word in sentence.split()]
        src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)  # [1, src_len]

        encoder_outputs, hidden = model.encoder(src_tensor)

        input = torch.tensor([tgt_vocab[SOS_TOKEN]]).to(device)
        translated = []

        for _ in range(max_len):
            output, hidden = model.decoder(input, hidden, encoder_outputs)
            top1 = output.argmax(1).item()

            if top1 == tgt_vocab[EOS_TOKEN]:
                break

            # 获取词汇表中对应的词
            translated_word = next((k for k, v in tgt_vocab.items() if v == top1), UNK_TOKEN)
            translated.append(translated_word)
            input = torch.tensor([top1]).to(device)

        return ''.join(translated)

if __name__ == "__main__":
    # 加载训练参数
    args = get_args()

    # 定义语料库文件路径
    src_file = os.path.join('.', args.src_corpus)
    tgt_file = os.path.join('.', args.tgt_corpus)

    # 读取中文和英文句子
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = f.readlines()

    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = f.readlines()

    # 确保两者长度相同
    assert len(src_sentences) == len(tgt_sentences), "源语言和目标语言句子数量不匹配"

    # 创建句子对，并去除空行
    sentence_pairs = [
        (en.strip(), cmn.strip()) for en, cmn in zip(src_sentences, tgt_sentences)
        if en.strip() and cmn.strip()
    ]

    # 如果指定了最大句子对数量，则进行截取
    if args.max_pairs is not None:
        if args.max_pairs < len(sentence_pairs):
            sentence_pairs = random.sample(sentence_pairs, args.max_pairs)
            print(f"已随机选择 {args.max_pairs} 句子对用于训练")
        else:
            print(f"指定的最大句子对数量 {args.max_pairs} 大于或等于总句子对数量。将使用全部句子对。")

    src_sentences, tgt_sentences = zip(*sentence_pairs)

    # 构建词汇表
    src_vocab = build_vocab(src_sentences, is_target=False)
    tgt_vocab = build_vocab(tgt_sentences, is_target=True)

    # 创建数据集和数据加载器
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # 模型参数
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(tgt_vocab)
    ENC_EMB_DIM = args.enc_emb_dim
    DEC_EMB_DIM = args.dec_emb_dim
    HID_DIM = args.hid_dim
    NUM_EPOCHS = args.num_epochs

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
    attention = Attention(HID_DIM)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    PAD_IDX = tgt_vocab[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 训练模型
    train_model(model, dataloader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)

    # 保存模型和词汇表
    save_model(model, src_vocab, tgt_vocab, path=args.model_path, vocab_path=args.vocab_path)

    # 进行翻译测试
    test_sentence = "I will try it."
    translation = translate(test_sentence, model, src_vocab, tgt_vocab, device)
    print(f"Translation Result: {translation}")