import sys
import NB 
import TAN

def usage_message():
	print "Usage: ./bayes.sh <train_file> <test_file> <n|t> "

if __name__ == '__main__':
	#print sys.argv[1]
	#print sys.argv[2]
	#print sys.argv[3]

	if (not len(sys.argv) == 4) or (sys.argv[3] != 'n' and sys.argv[3] != 't'):
		usage_message()
		exit()

	trainFile = sys.argv[1]
	testFile = sys.argv[2]
	algo = sys.argv[3]
	if algo == 'n':
		# Call NB 
		NB.bayes_learning(trainFile,testFile)
	elif algo == 't':
		TAN.tan_learning(trainFile,testFile)

