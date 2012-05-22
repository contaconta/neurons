import commands
from isNumber import *

def GetFloatingPointNumberFromExecOutput(command):
    output = commands.getoutput(command)
    words = output.split()
    numberArray = []
    for word in words:
        if IsNumber(word):
            numberArray.append( float(word) )
    return numberArray