'''
AI Coding HW

Ivan Christian
'''
from search_py3.search import Problem


WORDS = set(i.lower().strip() for i in open('words2.txt'))
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'



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
			oldLetter = oldWord[idx].lower()
			for newLetter in ALPHABET:
				if (newLetter != oldLetter):
					newWord = oldWord.replace(oldLetter, newLetter, idx+1)
					if((newWord is not None) and (is_valid_word(newWord))):
						newWords.append(newWord)

		return newWords

	def result(self, state, actions):
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
	# start = input('Please input a valid word to start the word ladder : ')
	# end = input('Please input a valid word to end the word ladder : ')
	start = 'best'
	end = 'math'

	if is_valid_word(start) and is_valid_word(end):
		wl = Word_Ladder(start, end)
		newWords = wl.actions(start)
		check = wl.goal_test(end)

		print(newWords)
		print(check)
	else:
		pritn('Not a valid word')




if __name__ == '__main__':
	run()