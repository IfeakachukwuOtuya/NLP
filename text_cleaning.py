
import nltk 

from nltk.stem import PorterStemmer #this will help to find the root word

stemmer = PorterStemmer()

#from nltk.stem import WordNetLemmatizer # this give the complete or proper word
#lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords  

paragraph = '''Customer chunks and customer segmentation are both related to organizing and understanding a business’s customer base, but they serve slightly different purposes.

In simple terms, customer segmentation refers to the process of dividing a company’s customers into distinct groups based on shared characteristics such as demographics, buying behavior, interests, or value to the business. This helps businesses tailor marketing, service, and product strategies to better meet the needs of each group. On the other hand, customer chunks typically refer to larger, broader categories or blocks of customers that are grouped together for high-level analysis — often based on major revenue contributions, geography, or lifecycle stage. While segmentation is usually data-driven and fine-grained, chunking may be more strategic or operational, helping companies quickly identify and prioritize key customer groups.'''



# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)


#words = nltk.word_tokenize(paragraph)


#Using for loop for all of sentence and using word_tokenize will convert all sentences
#Basicallly i am writting for word in words and i am taking from unique word from stopword
#Stemming

for  i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
    
    
    
    
    
    
    
    