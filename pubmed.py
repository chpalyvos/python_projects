## PROJECT PUBMED SEARCH
def extraction(text,arg):
# A function that returns the title or the abstract of a text separated in its lines, in form:
# (text)
# TI - (text)
# (text)
# AB - (text)
# (text)
# Obviously, in order to extract the title and the abstract, we hold the text next to 'TI' and 'AB', respectively
	res=[] # the list to return
	for k in range(len(text)): # run through the lines
		line = text[k]
		if line[0:2]==arg: # if found
			res.append(''.join(line.split('-')[1:])) # append the rest of the line (after the '-') to the res list
								 # join function is used because the line is in list form
			kk=k+1 # Now, append the lines that don't belong to another sector 
			next_line=text[kk]
			while (not(next_line[0:2].isupper())): # the lines that belong to another sector, begin with 2 UPPERCASE characters
				res.append(next_line)
				kk+=1
				next_line=text[kk]
			break	
	return res

def representsInt(s):
# A function that decides whether a string represents an integer 
	try: 
		int(s)
		return True
	except ValueError:
		return False

def daysFeb(year):
	# A function tha returns the maximum days of February depend on the year.
	# Remeber that....for example:
	# 2016 --> 29 , but 2100 --> 28, buutttt 2400 --> 29!!
	if year%4==0:
		if year%100==0:
			if year%400==0:
				return 29
			else:
				return 28
		else:
			return 29
	else:
		return 28

def maxDayofMonth(month,year):
	# A function that returns the maximum days a month has.
	# Remember that the maximum days February has, depend on the current year
	if month == 1:
		return 31
	elif month==2:
		return daysFeb(year)
	elif month==3:
		return 31
	elif month==4:
		return 30
	elif month==5:
		return 31
	elif month==6:
		return 30
	elif month==7:
		return 31
	elif month==8:
		return 31
	elif month==9:
		return 30
	elif month==10:
		return 31
	elif month==11:
		return 30
	elif month==12:
		return 31

def giveValidDate(date_form):
	# A function that returns the dates typed by the user. 
	month=''
	day=''
	while True:
		year=raw_input('Year (4-digits): ') # type the 4-digit year
		if representsInt(year) and len(year)==4 and int(year)<=2017 and int(year)>=0: # if in correct form
			if date_form >1: # if said that wants months
				while True:
					month=raw_input('Month of year '+str(year) +' (2-digits): ') # type the 2-digit month
					if representsInt(month) and len(month)==2 and int(month)<=12 and int(month)>=1: # if in correct form
						year=year+'/'
						if date_form >2: # if said that wants days
							while True:
								day=raw_input('Day of month '+str(month) +' (2-digits): ') # type the 2-digit day
								if representsInt(day) and len(day)==2 and int(day)<=maxDayofMonth(int(month),int(year[:-1])) and int(day)>=1: # if in correct form
									month=month+'/'
									break
								else:
									print 'Invalid answer entered. PLZ read again and answer in a correct form.'
						break
					else:
						print 'Invalid answer entered. PLZ read again and answer in a correct form.'
			break
		else :
			print 'Invalid answer entered. PLZ read again and answer in a correct form.'
	return year + month + day

from Bio import Entrez
import sys
import csv

Term = raw_input('Enter the term you would like to search for: ') # the user types his query

while True: # until the input is correct
	dates=raw_input('Would you like to search your query within special dates?? Type y OR n and hit ENTER: ') # the user says if he wants to limit his search in a specific time  
	if dates=='y':
		want_dates=True
		break
	elif dates=='n':
		want_dates=False
		break
	else :
		print 'Wrong answer entered. PLZ read again and answer in a correct form. : )'
if want_dates: # if he said yes, he has to give the dates
	print 'As for the dates, you can enter either 1)the years or 2)the years and months or 3)the years, the months and the days.'
	while True: # until the input is correct
		# 3 forms of dates given:  1)YYYY 2)YYYY/MM 3) YYYY/MM/DD
		date_form=raw_input('Choose one form between 1 , 2 , 3: ')
		if representsInt(date_form) and int(date_form)<4 and int(date_form)>0: # if he typed '1','2' or '3'
			date_form = int(date_form)
			print 'Giving Starting Date . . . '
			startingDate	=	giveValidDate(date_form)		# give the starting date in the form you said		
			print 'Giving End Date . . . '
			endDate				=	giveValidDate(date_form)		# give the end date in the form you said
			break
		else :
			print 'Invalid answer entered. PLZ read again and answer in a correct form.'

Entrez.email = "xristos.776@gmail.com"     		# Always tell NCBI who you are
if want_dates==True:
	handle = Entrez.esearch(db='pubmed', 						# database to search at
													term=Term,							# the term to search for
													retmax=10000000,				# maximum results of our query
													# if we want to find how many are the results of the current search we have to execute this command and the next one, once before to take the number , so it doesn't make any sense.
													datetype='pdat', 				# sort by publication date
													mindate=startingDate, 	# set the starting date
													maxdate=endDate)				# & the end date of publication 
else:
	handle = Entrez.esearch(db='pubmed', 			
													term=Term,			
													retmax=10000000	,			
													datetype='pdat')
	

record = Entrez.read(handle)				# the dictionary with details about the search results
id_list=record['IdList']				# the information that concerns us is the list with the Pub ID's

# An easy way to write a word in each cell of a line is to have a string of words separated by a semicolon ';'
# So the title & abstract have to be in a form 'word1;word2;...;wordk'
with open('titles_abstracts.csv','a') as f:
	for pid in id_list: # for every result (article)
		handle = Entrez.efetch( db='pubmed',            # fetch the articles in database pubmed
														id=pid,									# that their PID belongs to our PID list
														retmode='text',					# data format of records returns --> text
														rettype='medline') 		 	# record view returned (we choose meline to extract easier our data)
		
		x = handle.read()				# the large string that represents the text that describes the information below
		x_spl=x.split('\n')			# separate in lines to process easier
		
		# title & abstract extracted via the function 'extraction'
		# extraction returns the title & abstract as a list of strings with useless gaps
		title=''.join(extraction(x_spl,'TI'))		
		abstract=''.join(extraction(x_spl,'AB'))	# ''.join() --> convert this list of strings to 1 string(concatanate everything in the list)
		title_in_words=title.split()			# Separate into words (disappear the gaps)
		abstract_in_words=abstract.split()		# .split() --> .split(' ') --> separate strings that there is gap between them

		title_as_str=';'.join(title_in_words)		# Now the have the words, concatanate the list of worss to one unified string with
		abstract_as_str=';'.join(abstract_in_words)	# one semicolon between them
		# Write to the CSV file
		# 1st line for each article 'PUID: ....'
		f.write('PUID:'+str(pid)+'\n')
		# 2nd line: 'title:'
		f.write('title:\n')
		# 3rd line : '.....'(the title)
		f.write(title_as_str)
		f.write('\n')
		# 4th line : 'abstract:'
		f.write('abstract:\n')
		# 5th line : '.....'(the abstract)
		f.write(abstract_as_str)
		f.write('\n')
