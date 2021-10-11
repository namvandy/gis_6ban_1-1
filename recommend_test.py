import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

# from numpy import dot
# from numpy.linalg import norm
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(USE_CUDA)

tokenizer = torch.load(r'C:\Users\namva\자연어\koonggwangju\gis_6ban_1-1\tokenizer.pth', map_location=device)
bert = torch.load(r'C:\Users\namva\자연어\koonggwangju\gis_6ban_1-1\bert.pth', map_location=device)


train = pd.read_csv('C:/Users/namva/자연어/dataset/감성대화변환/train.csv')
train = train.loc[:200]

def Build_X (sents, tokenizer, device):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1).to(device)
# N, 3
# 데이퍼프레임에 데이터 추가
def data_add(df,sentence):
    new_data = [sentence,0] # 이거슨...컬럼 위치찾기..
    df.loc[len(df)] = new_data
# 입력 df # sentence는 입력할 문장
def text_recommend(df, sentence):
    data_add(df, sentence) # 데이터 추가
    docs = df[df.columns[0]] # person['사람문장1']
    sen = [sent for sent in docs] # 안에 있는 문장들 리스트화
    x = Build_X(sen, tokenizer, device) # 토크나이저 함수 적용
    input_ids = x[:, 0] # ids 추출
    token_type_ids = x[:, 1]
    attention_mask = x[:, 2]
    H_all = bert(input_ids, token_type_ids, attention_mask)[0]
    H_cls = H_all[:, 0, :]
    input_ids_numpy = H_cls.detach().cpu().numpy()
    # matrix = np.asmatrix(input_ids_numpy)
    matrix = np.array(input_ids_numpy) # ids matrix화
    cosine_matrix = cosine_similarity(matrix, matrix) # 코사인유사도 적용
    # cosine_matrix = cos_sim(matrix, matrix)
    #cosine_matrix = F.cosine_similarity(matrix, matrix).to(device)
    np.round(cosine_matrix, 4)
    idx2sent = {} # index 입력 -> 문장 출력
    for i, c in enumerate(docs): idx2sent[i] = c
    sent2idx = {} # 문장 입력 -> index 출력
    for i, c in idx2sent.items(): sent2idx[c] = i
    idx = sent2idx[sentence] # 문장을 넣어 index 출력
    sim_score = [(i, c) for i, c in enumerate(cosine_matrix[idx]) if i != idx]
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = [(idx2sent[i], c) for i, c in sim_score[:10]]
    return print(sim_score)
text_recommend(train,'오늘 점심을 먹고 난 후 부터 아무것도 먹지 못했다. 오늘 점심은 짜장면이었다. 지금은 새벽 3시다. 졸리다. 눕고 싶다. 언제쯤 끝날까. 13시간 째 컴퓨터 앞에 앉아 있다')

### 2021-10-12 3:32 Mr.Gang : 미치겠다.
# 데이터를 기존 10000개에서 200개만 넣었따. 이 이상넣으면 gpu가 가버린다. 컴퓨팅 파워가 부족하다.
# 문맥을 파악해서 비슷한 문맥의 문장을 추전해주고 있다.
# 데이터와 컴퓨팅 파워가 문제가 있어서 한계가 있다. 충분한 데이터와 컴퓨팅파워가 있다면 훌륭한 추천이 가능하다.