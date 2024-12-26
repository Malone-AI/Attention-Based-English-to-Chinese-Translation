import pickle
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from config import get_args

# 导入模型相关类和函数
from demo import Seq2Seq, Encoder, Attention, Decoder, translate

app = Flask(__name__)

# 特殊标记
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab['src_vocab'], vocab['tgt_vocab']

def load_model(model_path, vocab_path, device):
    src_vocab, tgt_vocab = load_vocab(vocab_path)
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(tgt_vocab)
    ENC_EMB_DIM = 300
    DEC_EMB_DIM = 300
    HID_DIM = 768
    
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
    attention = Attention(HID_DIM)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, src_vocab, tgt_vocab

# 配置参数
args = get_args()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和词汇表
model, src_vocab, tgt_vocab = load_model(args.model_path, args.vocab_path, device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    english_text = data.get('text', '').strip()
    if not english_text:
        return jsonify({'error': '未提供翻译文本'}), 400

    try:
        chinese_translation = translate(english_text, model, src_vocab, tgt_vocab, device)
        return jsonify({'translation': chinese_translation})
    except Exception as e:
        print(f"翻译错误: {e}")
        return jsonify({'error': '翻译失败'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)