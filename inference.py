import torch
from utils.tokenizer import basic_tokenizer
from model.transformer_base import transformer_base
from utils._utils import preprocess
from vocabulary import Vocab
class dialogue_bot:
    def __init__(self,vocab_path,model_ckpt_path,model_params,chat_max_length):
        self.tokenizer=basic_tokenizer(vocab_path)
        self.model=transformer_base(self.tokenizer.vocab_size,model_params['embed_dim'],model_params['nheads'])
        self.model.load_state_dict(torch.load(model_ckpt_path))
        self.chat_max_len=chat_max_length

    def nextsent(self,query):
        query=preprocess(query)
        if not query:
            return '无效输入！'
        src_ids=self.tokenizer.tokenize(list(query))
        src = torch.tensor(src_ids).unsqueeze(0)
        tgt_ids=self.tokenizer.tokenize(['<bos>'])
        res= []
        max_length=self.chat_max_len
        cnt=0
        self.model.eval()
        while cnt<max_length:
            tgt=torch.tensor(tgt_ids).unsqueeze(0)
            out=self.model(src,tgt)
            cur=out[-1,0,:].argmax().item()
            if self.tokenizer.detokenize([cur])[0]=='<eos>':
                break
            tgt_ids.append(cur)
            res.append(cur)
            cnt+=1
        return ''.join(self.tokenizer.detokenize(res)) if res else False
if __name__=="__main__":
    vocab_path='./data/basic_vob.txt'
    ckpt_path='./ckpt/trans_xhj_v2_steps_35000.pkl'
    model_params={"embed_dim":300,"nheads":15}
    generate_bot=dialogue_bot(vocab_path,ckpt_path,model_params,50)
    generation=[]
    generate_proces='段誉听到了背后脚步声音，待要回头，右肩已被抓住'
    for i in range(6):
        query=generate_proces
        generate_proces =generate_bot.nextsent(query)
        generation.append(generate_proces)
    print(generate_proces)

