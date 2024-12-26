import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Translation Model Parameters")
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--enc_emb_dim', type=int, default=256, help='编码器嵌入维度')
    parser.add_argument('--dec_emb_dim', type=int, default=256, help='解码器嵌入维度')
    parser.add_argument('--hid_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--model_path', type=str, default='model.pth', help='模型保存路径')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='词汇表保存路径')
    args = parser.parse_args()
    return args