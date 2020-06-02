'''
AI Coding HW

Ivan Christian
'''
from search_py3.search import Problem

def is_valid_word(word):
	'''
	Check 

	Args:
	- word : str (word for checking)

	Return :
	- check : bool (whether word is in the list of words)
	'''
	check = word.lower() in WORDS
	return check


class Word_Ladder(Problem):
	'''
	class inheritting the Problem from search
	'''
	def actions(self, state):
		'''
		action to be taken

		Args:
		- state : str (old word)
		return:
		- newWords : list (list of the subsequent waords)
		'''
		newWords = []
		oldWord = state

		for idx in range(len(oldWord)):
			oldLetter = oldWord[index].lower()
			for newLetter in ALPHABET:
				if (newLetter != oldLetter):
					newWord = oldWord.replace(oldLetter, newLetter, index+1)
					if((newWord is not None) and (is_valid_word(newWord))):
						newWords.append(newWord)
		return newWords
	def result(self, state, action):
		pass
	def value(self, state):
		pass



def run():
	'''
	You may ﬁnd the constants and functions in Python's string module helpful;
	you need to import string if you want to use it. Here are some simple test cases:
	"cars" to "cats"
	"cold" to "warm"
	"best" to "math"
	'''

	WORDS = set(i.lower().strip() for i in open('words2.txt'))
	ALPHABET = 'abcdefghijklmnopqrstuvwxyz'




if __name__ == '__main__':
	run()