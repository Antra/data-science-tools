# Lambda is the anonymous function in Python; not to be confused with AWS lambda functions
# They are called anonymous functions, because they don't maintain external (global) state - and return new data instead of manipulated data
from functools import reduce
x = ['Python', 'programming', 'is', 'awesome!']
print(sorted(x))
# sort by case insensitive (e.g. same as key=str.lower)
print(sorted(x, key=lambda arg: arg.lower()))
# sort by second character
print(sorted(x, key=lambda arg: arg[1]))


# filter
x = ['Python', 'programming', 'is', 'awesome!']
print(list(filter(lambda arg: len(arg) < 8, x)))
# NB, also calling list because filter() is also returning an iterable; so filter() would just return the values one by one; list() forces them all in memory at once


# map
x = ['Python', 'programming', 'is', 'awesome!']
print(list(map(lambda arg: arg.upper(), x)))


# reduce
x = ['Python', 'programming', 'is', 'awesome!']
print(reduce(lambda val1, val2: val1 + val2, x))
# reduce() is part of the functools package
