### TEXT MINING
# Give a word and we ll see how many times it appears in the csv file
import csv

flag=True
while flag: # the user can enter as many words as he wants
	keyword = raw_input('Give a word to search in the csv file with all the information about your search: ')
	with open('titles_abstracts.csv','r') as f:
		r = csv.reader(f,delimiter=';',quotechar='|')
		count=1 # a counter that presents the index of line is processed
		freq=0 # the frequency of the key word
		for row in r: 
			# we obviously wish to search for the keyword in the title and the abstract. 
			# title is found in the 3rd line of each article and the abstract in the fifth.
			# Each article consists of 5 lines.
			# We also have to check if there is title or abstract.
			# Remember that in a few articles either title or abstract doesn't exist in PUBMED database.
			if len(row)>0 and (count%5==0 or count%5==3):	
				for word in row: # check each word separately
					if len(word)>1:
						word="".join(c for c in word if c not in ('!','(',')',':','[',']',':','"','?',';')) # punctuation except '.' has to be removed
						if word[-1]=='.': # '.' has to be removed only if it is in the end of the word because it means the end of the sentence
								word=word[:-1] # hold the word except the '.' if it is the last characher
						if '/' in word: # it might means that there are 2 words. For instance, 'Cancer/Alzheimer'
							word0=word.split('/')[0] # separate the 2 words
							word1=word.split('/')[1]
							# the match has to be case insensitive. So we check the lowercase of the 2 words
							if word0.lower()==keyword.lower() or word1.lower()==keyword.lower(): # check if one of them matches our keyword
								freq+=1
						else :	# if there is no '/' in the word:
							if word.lower()==keyword.lower():
								freq+=1
					elif len(word)==1 : # just a character to check
						if word.lower()==keyword.lower():
							freq+=1
					else : # empty cell, nothing has to be checked
						continue	
			count+=1 # next line
		print 'Keyword "'+keyword+'" appeared '+str(freq)+' times in csv'
		while True: # the user can decide if he wants to give another word to search for
			answer=raw_input('Do you wish to search for the frequency of another word???? Type y OR n: ')
			if answer=='y':
				break
			elif answer=='n':
				flag=False
				break
			else: 
				print 'Incorrect answer given, please read again and answer!'
