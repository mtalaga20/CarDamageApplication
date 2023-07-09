import math

def cross_entropy(predicted_val:int, actual_val:int) -> int:
    return -((actual_val * math.log(predicted_val) + (1-actual_val) * math.log(1-predicted_val)))



#Validation - print(cross_entropy(0.6, 1))