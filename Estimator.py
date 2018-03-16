result = 62.137

def kilometers_to_miles(kilometers, coeff):
    return kilometers * coeff

dopusk = 0.001
coeff = 123142
increment = 0.1

while abs(kilometers_to_miles(100, coeff) - result) > dopusk:
    coeff += increment
    increment = abs(kilometers_to_miles(100, coeff) - result)*dopusk
    print(coeff, increment)
