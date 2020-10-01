import pandas as pd
import jieba
from gensim.models import Word2Vec,KeyedVectors
import numpy as np
import time
from typing import List, Dict
import re
from collections import defaultdict
from tqdm import tqdm
import math
import os
import pickle

class SearchEngine:
	
	def __init__(self, documents: List[str], inverted_index: Dict[str, Dict[int, int]], 
				 k: float, b: float):

		self.docs =  [doc.split() for doc in documents]
		self.k = k
		self.b = b
		self.inverted_index = inverted_index
	
	def idf(self, word) -> float:

		num_docs = len(self.docs)
		num_contain_word_docs = 0
		for doc in self.docs:
			if word in doc:
				num_contain_word_docs += 1
		idf_weight = math.log2(
			(num_docs + 1) / num_contain_word_docs
		)
		return idf_weight
	
	def tf(self, word: str, query: List[str], relevant_doc: List[str], 
		   avg_docs_len: int) -> float:

		cwd = relevant_doc.count(word)
		cwq = query.count(word)
		numerator = (self.k + 1) * cwd
		denominator = cwd + self.k * (1 - self.b + self.b * (len(relevant_doc) / avg_docs_len))
		tf_weight = cwq * (numerator / denominator)
		return tf_weight
	
	def rank_by_bm25(self, query: List[str], relevant_docs: List[List[str]]) -> List[str]:

		scores = []
		unique_words = set(query)
		avg_docs_len = sum(len(doc) for doc in self.docs) / len(self.docs)
		for i, doc in enumerate(relevant_docs):
			doc_score = .0
			match_words = unique_words & set(doc)
			for word in match_words:
				doc_score += self.idf(word) \
				* self.tf(word, query, doc, avg_docs_len)
			scores.append((doc_score, i))
		scores.sort(reverse=True)
		ranked_docs = [''.join(relevant_docs[i]) for _, i in scores]
		return ranked_docs
	
	def accumulate(self, query, accumulator):

		for word in query:
			for doc_id, word_count in self.inverted_index[word].items():
				accumulator[int(doc_id)] += word_count
	
	def search(self, query, topk=5) -> List[str]:

		words = list(jieba.cut(query))
		accumulator = [0] * len(self.docs)
		self.accumulate(words, accumulator)
		accumulator = [(score, idx) for idx, score in enumerate(accumulator)]
		accumulator.sort(reverse=True)
		relevant_docs = [self.docs[idx] for _, idx in accumulator[:topk]]
		return self.rank_by_bm25(words, relevant_docs)

class bot:
	"""docstring for bot"""
	def __init__(self):
		print("agent init...")
		self.question_answer=dict()
		self.word2vec=None
		# self.questions=None
		self.normalizer=None
		self.cluster=None
		self.se1=None
		self.se2=None

		data=pd.read_csv("./data.csv")
		data.dropna(inplace=True)
		for k,v in zip(data['question'],data['answer']):
			self.question_answer[k]=v
		questions=data['question'].tolist()
		questions=list(map(lambda x:self.cut(x),questions))
		print("corpus len:",len(questions))

		##加载词向量模型
		t1=time.time()
		print('loading word vector model...')
		# model_file_name='D:\\BaiduNetdiskDownload\\word2vec_model_chinese\\baike_26g_news_13g_novel_229g.bin'
		model_file_name='./w2v_model.bin'
		self.word2vec=KeyedVectors.load_word2vec_format(model_file_name,binary=True)
		print('word vector model load success,shape:',self.word2vec['查询'].shape)
		t2=time.time()
		print(f'use time:{t2-t1}')

		##聚类并保存结果
		if not os.path.exists('./k_label.pkl'):
			sentence_vecs=self.get_corpus_vec(questions)
			print("feature vec len:",len(sentence_vecs))

			k_labels,cluster,normalizer=self.kmeans(sentence_vecs,2)
			with open('k_label.pkl','wb') as file:
				pickle.dump(k_labels, file)
			with open('cluster.pkl','wb') as file:
				pickle.dump(cluster, file)
			with open('normalizer.pkl','wb') as file:
				pickle.dump(normalizer, file)
		else:
			with open('k_label.pkl','rb') as file:
				k_labels = pickle.load(file)
			with open('cluster.pkl','rb') as file:
				cluster = pickle.load(file)
			with open('normalizer.pkl','rb') as file:
				normalizer = pickle.load(file)

		self.normalizer=normalizer
		self.cluster=cluster
		##把原始文本聚2类
		class1=[]
		class2=[]
		for i in range(len(k_labels)):
			if k_labels[i]==0:class1.append(questions[i])
			if k_labels[i]==1:class2.append(questions[i])
		# print(len(class1),class1[:6])
		# print(len(class2),class2[:6])

		##倒排索引和搜索
		inverted_index1=self.build_inverted_index(class1)
		inverted_index2=self.build_inverted_index(class2)

		self.se1 = SearchEngine(class1, inverted_index1, k=1, b=0.75)
		self.se2 = SearchEngine(class2, inverted_index2, k=1, b=0.75)

	def search(self,user_question):
		#print(user_question)
		if user_question in self.question_answer:
			# print(self.question_answer[user_question])
			return self.question_answer[user_question]
		else:
			vec=self.get_sentence_embedding(user_question)
			vec=self.normalizer.transform(vec.reshape(1,-1))
			vec=vec.astype(float)
			y_pred=self.cluster.predict(vec)
			# print(y_pred)

			bm25_results=[]
			if y_pred[0]==0:
				for doc in self.se1.search(user_question, topk=30):
					bm25_results.append(doc)
			else:
				for doc in self.se2.search(user_question, topk=30):
					bm25_results.append(doc)

			# print(bm25_results[0])
			res='你是不是要问：'+bm25_results[0]+'\n\n'+self.question_answer[bm25_results[0]]
			return res

	def cut(self,x):
		return " ".join(jieba.lcut(x))
	##取中文
	def token(self,string):
		return re.findall(r'[\u4e00-\u9fa5]+', string)

	def kmeans(self,tfidf,num_class):
		from sklearn.preprocessing import Normalizer
		from sklearn.cluster import KMeans
		normalizer = Normalizer()
		scaled_array = normalizer.fit_transform(tfidf)

		# 使用K-Means, 对全量文档进行聚类
		kmeans = KMeans(n_clusters=num_class,random_state=2,n_jobs=4)
		k_labels = kmeans.fit_predict(scaled_array)
		# 查看每一类的文档数目
		k_labels=k_labels.tolist()
		for n in range(num_class):
			print(k_labels.count(n))
		return k_labels,kmeans,normalizer

	def get_sentence_embedding(self,sentence:str):
		word2vec=self.word2vec
		sentence_cut=self.cut(''.join(self.token(sentence)))
		sentence_vec = np.zeros_like(word2vec['查询'])
		##查找每个句子中的每个词的词向量相加求平均得到句子向量
		##如果没有对应的词向量就随机生成一个
		additional_wordvec = {}
		for word in sentence_cut.split():
			if word in word2vec.vocab:
				sentence_vec += word2vec[word]
			elif word in additional_wordvec:
				sentence_vec += additional_wordvec[word]
			else:
				additional_wordvec[word] = np.random.random(word2vec['查询'].shape)
		##词向量均值作为句子向量
		return sentence_vec / len(sentence)

	def get_corpus_vec(self,corpus):
		vecs=[]
		for i in corpus:
			vec=self.get_sentence_embedding(i)
			vecs.append(vec.tolist())
		return vecs

	##对每一类建立倒排索引然后bm25搜索
	def build_inverted_index(self,documents):
		inverted_index = defaultdict(dict)
		for i, doc in tqdm(enumerate(documents)):
			try:
				tokens = doc.split()
			except:
				print(i,doc)
			for token in set(tokens):
				# 更新当前 token 在当前文档中的出现次数
				# https://www.runoob.com/python/att-dictionary-update.html
				inverted_index[token].update({i: tokens.count(token)})
		return inverted_index

if __name__=='__main__':
	import time
	b=bot()
	t1=time.time()
	ans=b.search('怎么修改密码')
	t2=time.time()
	print(f'use time:{t2-t1}')
	print(ans)
