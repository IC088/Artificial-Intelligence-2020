# HW 2 Documentation

## Run

To run call ```python3 semi_magic.py```


## Task
1.  Implement a CSP that captures this problem.  Use the filesemi-magic.pyin  the  code  distribution.   It  also  imports  csp.py.   You  should  use  thevariable names in the image above in your CSP formulation.
2.   Experiment solving this problem with the various solution methods, rang-ing from pure backtracking and its various enhancements/heuristics suchas variable and value orderings, forward checking, etc.
3.    Look  at  the  number  of  assignments  attempted  by  each  algorithm,  that should  give  you  some  idea  of  the  effectiveness  of  the  methods  on  this problem.   Elaborate  on  your  findings  from  these  results,  i.e.,  what  you understand.


## Answer
2. ![](https://i.imgur.com/eQBkCXH.png)
3. The minimum number of assignment can be seen as 9 which is reached by pure backtracking, forward checking enhancement to backtracking, maintaining arc consistency (AC3), and LCV enhancement to backtracking. MRV and Min conflict may get the minimum that the rest of the enhancement and algorithm methods but they do not reach the minimum consistently. 
    *a.* For AC3 and Forward-checking, they would consistently reach the minimum number because of the small size and it is easy to do the possible values check. For LCV, it rules out the smallest number of values connected to the current variable by constraint which leads to the minimum number of assignment consistently. 
    *b.* For MRV, it does not consistently reach the minimum because it chooses teh variable with the least number of possibilities, which in this case does not really matter  and can randomly choose any at the start since they all hav ethe same number of possibilities (1,2,3). 
    *c.* Pure backtracking algorithm strangely enough, consistently reaches the minimum value for asssignment. This may be because of the small number of domains and variables that the pure backtracking method can consistently reach the minimum value for assignment.