'''
Flight itenary Q2

Ivan Christian
1003056
'''
from search_py3.search import *

class FlightSearch(Problem):
	'''
	Itinerary search for the flight
	'''
	def actions(self, state): # OK
		'''
		Actions taken in the flight search problem
		'''
		flightList = []
		for flight in flightDB:
			if (state[0] == flight.start_city) and (state[1] <= flight.start_time):
				flightList.append(flight)
		return flightList

	def result(self, state, action):

		nextState = (action.getEndCity(),action.getEndTime())
		return nextState

	def goal_test(self, state):  # OK
		if (state[0] == self.goal[0]):
			return(state[1] <= self.goal[1])

	def value(self, state):
		pass


class Flight:
	'''
	Skeleton class for Flight
	'''
	def __init__(self, start_city, start_time, end_city, end_time): 
		self.start_city = start_city 
		self.start_time = start_time 
		self.end_city = end_city 
		self.end_time = end_time
	def __str__(self): 
		return str((self.start_city, self.start_time))+' -> '+ str((self.end_city, self.end_time))
	__repr__ = __str__



	def getStartCity(self):
		return self.start_city

	def getStartTime(self):
		return self.start_time


	def getEndCity(self):
		return self.end_city

	def getEndTime(self):
		return self.end_time

	def flightMatch(self, start_city, start_time):
		'''
		Q2: Matches

		function to match the flights

		Args:
		- start_city : string (city input)
		- start_time : int (time input)

		return:
		- check : boolean (check whether the flight leaves from teh same city at a time later than start time)
		'''
		for flight in flightDB:
			if (flight.start_city == start_city) and (flight.start_time >= start_time):
				return True
		return False



	def find_itinerary(self): # get the start time, start city, end city and deadline from the defined self variable
		'''
		Q3: get the flight itinerary based on the end time/deadline
		'''
		search = FlightSearch(initial = (self.start_city, self.start_time), goal = (self.end_city, self.end_time))

		try:
			solution = breadth_first_search(search).path()
			solution_list = []
			for node in solution:

				solution_list.append(node.state)
			return solution_list
		except :
			return 'No Possible Paths'

	def find_shortest_itinerary(self):
		'''
		Q4: get the shortest path for the flights
		'''
		deadline = 0
		search = FlightSearch(initial = (self.start_city, 1), goal = (self.end_city, deadline))

		while not breadth_first_search(search):
			search = FlightSearch(initial = (self.start_city, 1), goal = (self.end_city, deadline))
			deadline += 1
		solution = breadth_first_search(search).path()
		solution_list = []
		for node in solution:

			solution_list.append(node.state)
		return solution_list



flightDB = [Flight('Rome', 1, 'Paris', 4), 
		Flight('Rome', 3, 'Madrid', 5), 
		Flight('Rome', 5, 'Istanbul', 10), 
		Flight('Paris', 2, 'London', 4), 
		Flight('Paris', 5, 'Oslo', 7), 
		Flight('Paris', 5, 'Istanbul', 9), 
		Flight('Madrid', 7, 'Rabat', 10), 
		Flight('Madrid', 8, 'London', 10), 
		Flight('Istanbul', 10, 'Constantinople', 10)]


def run():
	flight = Flight('Rome', 1, 'Oslo', 9)
	print(flight.find_itinerary())

	print(flight.find_shortest_itinerary()) # Does not care about the deadline since it is self imposed


if __name__ == '__main__':
	run()

