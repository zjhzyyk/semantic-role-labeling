def getfmeasure(f):
	dic = {}
	for i, line in enumerate(f):
		if "evaluation:" in line:
			arr = line.split()
			t = arr[0]
			dic[t] = float(f[i+4].split()[1])
	return dic

def readfile(f):
	ret = []
	for line in f:
		ret.append(line)
	return ret

def output(d,str):
	print "\n"+str
	for k, v in d.iteritems():
		if k in baseline:
			if abs(v-baseline[k])>1e-3:
				print k, ":", (v-baseline[k])*100, "%"

baseline = getfmeasure(readfile(open("baseline", "rb")))
output(getfmeasure(readfile(open("contextrelative", "rb"))), "contextrelative")
output(getfmeasure(readfile(open("headofppp", "rb"))), "headofppp")
output(getfmeasure(readfile(open("lemmaandhead", "rb"))), "lemmaandhead")
output(getfmeasure(readfile(open("lemmaandphrasetype", "rb"))), "lemmaandphrasetype")
output(getfmeasure(readfile(open("lemmaandppp", "rb"))), "lemmaandppp")
output(getfmeasure(readfile(open("partialpath", "rb"))), "partialpath")
output(getfmeasure(readfile(open("pathlen", "rb"))), "pathlen")
output(getfmeasure(readfile(open("ppnphead", "rb"))), "ppnphead")
output(getfmeasure(readfile(open("voiceandposition", "rb"))), "voiceandposition")
output(getfmeasure(readfile(open("withfirstlastword", "rb"))), "withfirstlastword")
