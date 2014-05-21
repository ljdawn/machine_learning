#coding:utf8
"""
====================
feature selection tool 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-11-05"""

__all__ = ['']

import sys
sys.path.append('..')
from toolbox import text_manufactory
from toolbox import text_chinese_filter
import cPickle as pickle
import gensim
from gensim import corpora, models, similarities

#text file path
path = '../testing/d_text_2_matrix'
#reading text file
f = open(path,'r').readlines()

#text to list_list
list_f = list(text_manufactory.text_2_list(f))

corpus = []
text2id = {}
id2text = {}
tmd = set([])
c_path = 'corpus/'
d_path = 'dict/'
fn = 'd'

dd,ld,dc,lc = 1,1,1,1
l = 2

if dd == 1:
        #---dump dict
        dictionary = corpora.Dictionary(list_f)
        dictionary.save(d_path+fn+'.dict')
if ld == 1:
        #---load corpus
        id2word = corpora.Dictionary.load(d_path+fn+'.dict')
if dc == 1:
        #---dump corpus
        for i in range(len(list_f)):
                tm = []
                for word in list_f[i]:
                        tmd.add(word)
        for data in enumerate(list(tmd)):
                text2id[data[1]] = data[0]
                id2text[data[0]] = data[1]
        for i in range(len(list_f)):
                tm = []
                for word in list_f[i]:
                        tm.append((text2id[word],1.0))
                corpus.append(tm)
        corpora.MmCorpus.serialize(c_path+'corpus.mm', corpus)
        corpora.SvmLightCorpus.serialize(c_path+'corpus.svmlight', corpus)
        corpora.BleiCorpus.serialize(c_path+'corpus.lda-c', corpus)
        corpora.LowCorpus.serialize(c_path+'corpus.low', corpus)
if lc == 1:
        #---load corpus
        m_corpus = corpora.MmCorpus(c_path+'corpus.mm')

lda = gensim.models.ldamodel.LdaModel(corpus = m_corpus,id2word=id2word, num_topics=l, update_every=1, chunksize=10000, passes=1)

for i in range(l):
        for item in lda.show_topic(i):
                print item[1],'|',
        print '\n'
