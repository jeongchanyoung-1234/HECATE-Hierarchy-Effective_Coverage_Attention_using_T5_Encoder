import mecab
from copy import deepcopy

import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    def __init__(self, train_data_path, valid_data_path, threshold=0.12):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.vectorizer = TfidfVectorizer()
        self.threshold = threshold
        self.tokenizer = mecab.MeCab()
        self.stopwords1 = ['출장', '조건', '업소', '애인', '업 소', '모르', '마담', '술집', '미성년자']
        self.stopwords2 = ['회사원', '자영업']
        self.stop_id = ['5314', '105212', '77377', '267', '129125', '1209', '105962', '100', '103486', '103779', '2405', '110301', '122406', '108785', '353s', '105026', '106295']
        
    def get_similarity(self, sent1, sent2):
        sent1 = ' '.join(self.tokenizer.morphs(sent1.split('<SEP>')[1]))
        sent2 = ' '.join(self.tokenizer.morphs(sent2.split('○')[1]))

        sentences = (sent1, sent2)
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        cos_similar = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        if cos_similar < self.threshold:
            for stopword in self.stopwords1:
                if stopword in sent1:
                    return cos_similar, True
            for stopword in self.stopwords2:
                if stopword in sent2:
                    return cos_similar, True
            print(sent1)
            print(sent2)
            print(cos_similar)
            return cos_similar, False
        return cos_similar, True

    def save(self, result:list, data_path):
        new_data_path = '.' + data_path.split('.')[1] + '_cln.' + data_path.split('.')[2]
        for r in result:
            with open(new_data_path, 'a') as f:
                f.write(str(r) + '\n')

    def preprocess(self):
        train_result = []
        train_total_cnt, train_new_cnt = 0, 0

        with open(self.train_data_path, 'r') as f:
            train_list = f.readlines()
        for l in train_list:
            train_total_cnt += 1
            dic = eval(l)
            
            if dic['고소장 id'] in self.stop_id:
                continue

            similarity, flg = self.get_similarity(dic['input'], dic['output'])
            if flg:
                train_result.append(dic)
                train_new_cnt += 1
            else:
                print('cut off ID {}'.format(dic['고소장 id']))
        self.save(train_result, self.train_data_path)

        print('[RESULT]')
        print('{} -> {}'.format(train_total_cnt, train_new_cnt))
        

if __name__ == '__main__':
    processor = Preprocessor('./data/Training_set_prostitution_only.json', './data/Valid_set_prostitution_only.json')
    processor.preprocess()
    # sent1 = '범죄 수법 : 성매매 유도 사기 <SEP> 피고소인 신분 : 출장안마하는 여자입니다 <SEP> 고소인 신분 : 일반 직장인입니다 <SEP> 날짜 : 2019. 10. 27. 08:53경 <SEP> 장소 : 채팅어플입니다. <SEP> 피고소인을 알게된 경위 : 어떤 여자가 채팅에서 성매매 한다고 저한테 접근했습니다. <SEP> 거래 방법 : 예약금으로 15만원 보내면 저랑 해주겠다했습니다 <SEP> 거짓말의 내용 : 15만원 보냈더니 추가금 30만원을 넣으라고하더라고요. 그래서 넣었더니 입금자명 추가금으로 안했다면서 100만원을 더 넣으라했어요. 100만원 더 넣으니까 환불받고싶으면 x2한 돈 넣으라고 하더라고요 <SEP> 재산 마련 방법 : 통장에 현금을 가지고 있었습니다. <SEP> 재산의 처분 방법 : 요구하는대로 입금하여 총 1150만원을  여러차례 걸쳐 입금했습니다. <SEP> 거짓임을 깨닫게 된 계기 : 지금도 570만원을 더보내라고 용역에 접수된다면서 협박을 하고 있으며 여자도 동시에 재촉연락을 하기에 사기라고 인식하게 되었습니다 <SEP> 다른 피해사실 : 저는 다른 피해자가 있는지에 대해서는 모릅니다. <SEP> 고소 이유 : 억울한 마음에 처벌을 시키고 싶고 빼앗긴 돈도 돌려받고자 고소를 하게 되었습니다. <SEP> 다른 민형사 : 없습니다.'
    # sent2 = '○ 피고소인은 출장 안마를 하는여성 입니다. ○ 고소인은 직장인 입니다.○ 2019-10-27경  고소인은 채팅 어플에서  채팅을 하다 피고소인을 알게 되어 대화를 하던 중  고소인이 피고소인의 계좌로 15만원을 송금 하면 피고소인은 고소인을 만나 성관계를  맺어  주기로 하였습니다. ○ 피고소인은 고소인에게 자신의 계좌로 15만원을 송금 받고 추가로 30만원 송금을 요구 하여 고소인이 이를 송금 하였습니다. 그러나 피고 소인은 고소인이 입금자명을 추가금이라 입력하지 않았다, 환불해 주겠 다, 라는 등 수회에 걸쳐 거짓말을 하며 계속 추가 송금을 요구 하였습 니다. 이에 속은 고소인은 피고소인의 말을 믿고 계속 추가 송금을 하여 합계 1150만원에 이르렀습니다. 그럼에도 불구하고 피고소인은 고소인 에게 환불해 주지 않고 추가 입금을 요구하며 고소인의 착오로 입금한 위 금원을 편취 하였습니다.○ 고소인은 예금으로 마련한 돈을  피고소인의 계좌로 이체송입금 하였습니다. ○ 결국, 고소인은 피고소인이 계속 추가 입금을 요구 하여 자신이 사기당했음을 깨닫게 되었습니다. ○ 고소인은 다른 피해는 없습니다.'

    # sent1 = '범죄 수법 : 성매매 유도 사기 <SEP> 피고소인 신분 : 대학생입니다 <SEP> 고소인 신분 : 저는 대학생입니다 <SEP> 날짜 : 2021-02-23 입니다. <SEP> 장소 : 소개팅어플입니다 <SEP> 피고소인을 알게된 경위 : 소개팅 어플로 대화를 하다가 너무 예쁜분이 있어서 대화를 하게됐습니다 <SEP> 거래 방법 : 가볍게 만나자했는데 자기는 등록금 버느라 돈이 없어서 만나줄시간이 없다했습니다. 그래서 제가 50만원 보내면 해줄수있냐했더니 그러겠다해서 바로 만나게됐습니다 <SEP> 거짓말의 내용 : 제가 50만원을 보냈더니 너무 미안하고 고마워라고 한마디만 남기고 사라졌습니다 <SEP> 재산 마련 방법 : 알바를 해서 모은 돈 입니다. <SEP> 재산의 처분 방법 : 현재 여성과 만나지도 못하였고 50만원을 입금한 상태입니다. <SEP> 거짓임을 깨닫게 된 계기 : 설마설마했는데 마지막 말 듣고 심장이 내려앉았습니다 <SEP> 다른 피해사실 : 없습니다. <SEP> 고소 이유 : 사람 마음을 가지고 논게 괘씸해서 고소하고싶습니다 <SEP> 다른 민형사 : 아니오. 따로 진행하고 있진 않습니다.'
    # sent2 = '○ 피고소인은 30대 남성으로 추측됩니다.○ 고소인은 가정주부 입니다.○ 2021-02-02. 오전에, 자택에서 고소인은 피고소인이 미국에 있는 고소인의 딸이 범죄를 저질러 합의금이 필요하다며 전화연락하였 습니다.   ○ 고소인이 피고소인이 알려준 계좌로 2,000만원을 입금하면 고소인 딸의 범죄를 잘 처리해주겠다며  유혹했습니다.  ○ 이에 고소인이 피고소인의 말을 믿고 그의 계좌로 2,000만원을 송금 한 뒤 딸에게 확인 전화를 하자 모든게 거짓인것으로 밝혀졌습니다. ○ 고소인은 적금을 해제하여 피고소인이 알려준 계좌에  2,000만원을 직접 입금했습니다.○ 그런데 고소인은 딸과 전화통화를 하고서 사기임을 인지하였습니 다. ○ 이런 보이스 피싱 사기는 처음 당해보고 다른 사람들의 피해에 대해 아는 것이 없습니다.'
    
    # mecab = mecab.MeCab()
    # sim = processor.get_similarity(sent1, sent2)
    # print(sim)