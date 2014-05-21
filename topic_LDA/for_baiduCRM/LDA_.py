from gensim import corpora, models, similarities
import gensim

texts = [line.strip().split(' ') for line in open('source_text/news_0','r').readlines()]

dictionary = corpora.Dictionary(texts)
#dictionary.save('dict/sample.dict')
print dictionary.token2id

id2word={2 :'money', 1:'loan', 3: 'river', 4:'stream', 0:'bank'}

corpus = [dictionary.doc2bow(text) for text in texts]

print corpus

lda = models.ldamodel.LdaModel(corpus=corpus, num_topics=2, id2word=id2word)

for topic in lda.show_topics(-1):
	print topic
for topics_per_document in lda[corpus]:
	print topics_per_document