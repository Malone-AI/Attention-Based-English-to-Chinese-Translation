import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Translation Model Parameters")
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--enc_emb_dim', type=int, default=300, help='编码器嵌入维度')
    parser.add_argument('--dec_emb_dim', type=int, default=300, help='解码器嵌入维度')
    parser.add_argument('--hid_dim', type=int, default=768, help='隐藏层维度')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--model_path', type=str, default='model.pth', help='模型保存路径')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='词汇表保存路径')
    parser.add_argument('--src_corpus', type=str, default='Tatoeba.cmn-en.en', help='源语言语料库路径')
    parser.add_argument('--tgt_corpus', type=str, default='Tatoeba.cmn-en.cmn', help='目标语言语料库路径')
    parser.add_argument('--max_pairs', type=int, default=None, help='最大句子对数量，用于限制训练数据量')
    args = parser.parse_args()
    return args