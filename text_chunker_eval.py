__author__ = 'John'
import nltk
from nltk.corpus import conll2000
from UnigramChunker import UnigramChunker
from BigramChunker import BigramChunker
from ConsecutiveNPChunker import ConsecutiveNPChunker

cp = nltk.RegexpParser("")
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
# print(cp.evaluate(test_sents))
# print(conll2000.chunked_sents('train.txt')[99])

print "RegEx Part of Speech chunker"
grammer = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammer)
print (cp.evaluate(test_sents))

test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

print "Unigram chunker"
unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

print "Bigram chunker"
bigram_chunker = BigramChunker(train_sents)
print(bigram_chunker.evaluate(test_sents))

print "Consecutve NP chunker with maxent classifier"
try:
    chunker = ConsecutiveNPChunker(train_sents)
    print(chunker.evaluate(test_sents))
except:
    print ("Error with maxext classifier. Is MaxEnt/Megam installed?")
