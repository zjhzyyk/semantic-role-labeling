from bllipparser import RerankingParser, tokenize
print "start loading model..."
rrp = RerankingParser.load_unified_model_dir('/home/yukang/selftrained')
print "finish loading model"
inputfile = "wsjtest"
outputfile = "wsjtest.reparse"
count = 0
data = open(inputfile)
output = open(outputfile, 'w')
sentence = []
for line in data:
	if len(line.split())==0:
		if len(sentence)==0:
			continue
		count+=1
		print "start solving", count
		#last line of the file must be a blank line to terminate the last sentence.
		l = [word[0].replace("(", "-LRB-").replace(")", "-RRB-") for word in sentence]
		ans = rrp.parse(l)
		output.write(str(ans[0].ptb_parse)+"\n")
		# if count > 1:
			# break
		sentence = []
	else:
		parts = line.split()
		sentence.append(parts)
output.close()