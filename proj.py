import tarfile
import cPickle as pickle
from sklearn.linear_model import SGDClassifier
from collections import deque
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.sparse import lil_matrix

class node:
	def __init__(self, parent, typ, start, end):
		self.type = typ
		self.children = []
		self.parent = parent
		self.start = start
		self.end = end
		self.i = -1
		self.headNP = None

	def addChild(self, child):
		child.parent = self
		child.i = len(self.children)
		self.children.append(child)

	def leftSibling(self):
		if self.parent==None:
			return None
		if self.i>0:
			return self.parent.children[self.i-1]
		return None

	def rightSibling(self):
		if self.parent==None:
			return None
		if self.i>=0 and self.i<len(self.parent.children)-1:
			return self.parent.children[self.i+1]
		return None

def preorderPrint(n, str):
	print str+n.type
	for child in n.children:
		preorderPrint(child, str+" ")

def prune(verb):
	candidates = []
	punc = {".", ":", "-LRB-", "-RRB-", ",", "--", "''", "``"}
	tmp = verb.parent
	vp = [verb]
	while tmp.type!="S1" and tmp.type!=None:
		vp.append(tmp)
		tmp = tmp.parent
	tmp = verb.parent
	while tmp.type!="S1" and tmp.type!=None:
		# print "cur node", tmp.type
		for i, child in enumerate(tmp.children):
			if child not in vp and child.type not in punc:
				if child.type=="NP" and (i+1)<len(tmp.children) and tmp.children[i+1].type=="PP" \
					and (tmp.children[i+1].end<verb.start or child.start>verb.end):
					newnode = node(tmp, "NP", child.start, tmp.children[i+1].end)
					newnode.children.append(child)
					newnode.children.append(tmp.children[i+1])
					if newnode.start==tmp.start and newnode.end==tmp.end and tmp.parent!=None:
						newnode.parent = tmp.parent
					candidates.append(newnode)
				if child.type=="PP":
					for chid in child.children:
						if chid.type=="NP":
							candidates.append(chid)
				if child.type=="S" or child.type=="SBAR":
					q = deque()
					q.append(child)
					while q:
						chd = q.popleft()
						candidates.append(chd)
						if chd.type=="S" or chd.type=="SBAR":
							for chid in chd.children:
								q.append(chid)
				else:
					candidates.append(child)
		tmp = tmp.parent
	return candidates

def getWords(sentence):
	words = []
	for word in sentence:
		words.append(word[0])
	return words

def getTags(sentence):
	tags = []
	for word in sentence:
		tags.append(word[1])
	return tags

def genParseStr(sentence, tags):
	ret = ""
	treeIdx = 2
	wordc = 0
	toSentIdx = {}
	for word in sentence:
		ret += word[treeIdx]
	i = 0
	while i < len(ret):
		if ret[i] == "*":
			toSentIdx[i+1] = wordc
			ret = ret[:i] + "(" + tags[wordc] + ")" + ret[i+1:]
			i += (2+len(tags[wordc]))
			wordc += 1
		else:
			i += 1
	return (ret, toSentIdx)

def findParenth(s):
	pos = {}
	stack = []
	s = s.replace("(()", "(#)")
	s = s.replace("())", "(#)")
	for i,c in enumerate(s):
		if c=="(":
			stack.append(i)
		elif c==")":
			pos[stack.pop()] = i
	return pos

def construct(sentence):
	# print "in construct"
	words = getWords(sentence)
	tags = getTags(sentence)
	s, strToSent = genParseStr(sentence, tags)
	# print s
	# print strToSent
	parenth = findParenth(s)
	# print parenth
	return buildTree(s[1:-1], parenth, strToSent, 1)

def construct2(parse):
	parenth = findParenth(parse)
	# print parenth
	i = 0
	l = len(parse)
	count = 0
	strToSent = {}
	while i<l:
		if parse[i]=="(":
			# print i, parenth[i]
			if "(" not in parse[i+1:parenth[i]]:
				strToSent[i+1] = count
				count += 1
		i+=1
	# print strToSent
	return buildTree(parse[1:-1], parenth, strToSent, 1)

def buildTree(s, parPos, str2sent, start):
	# print "s is", s
	# print "start is", start
	idx = []
	i = 0
	l = len(s)
	while i<l:
		if s[i]=="(" and s[i-1:i+2]!="(()" and s!="(":
			idx.append(i)
			i = parPos[i+start]-start+1
		else:
			i+=1
	l = len(idx)
	if l==0:
		#recur end
		# return node(None, s, str2sent[start], str2sent[start]) #for training set
		return node(None, s.split()[0], str2sent[start], str2sent[start]) # for reparse file
	else:
		ret = node(None, s[:idx[0]].rstrip(), 0,0)
		for ii in idx:
			child = buildTree(s[ii+1:(parPos[ii+start]-start)], parPos, str2sent, ii+start+1)
			ret.addChild(child)
		ret.start = ret.children[0].start
		ret.end = ret.children[-1].end
		return ret

def pruneFile(inputfile):
	count = 0
	data = open(inputfile)
	parsefile = open(inputfile+".reparse")
	sentence = []
	prunefp = 0.0
	prunefn = 0.0
	prunetp = 0.0
	for line in data:
		if len(line.split())==0:
			print "start solving"
			if len(sentence)==0:
				continue
			#last line of the file must be a blank line to terminate the last sentence.
			count+=1
			# if count > 1:
				# break
			ans = getPropAns(sentence)
			# root = construct(sentence)
			parse = parsefile.readline()[:-1]
			# print parse
			root = construct2(parse)
			words = getWords(sentence)
			# print ans
			printAns(ans, words)
			# preorderPrint(root, "")
			tfn, tfp, ttp = bfs(root, words, ans) #labeling verbs in the order of importance
			prunefn += tfn
			prunefp += tfp
			prunetp += ttp
			#bfs2(root, words, ans) #labeling verbs according their orders in the sentence
			sentence = []
		else:
			parts = line.split()
			sentence.append(parts)
	printEval(prunefn, prunefp, prunetp, "prune evaluation:")

def printAns(ans, words):
	for verbi, prop in ans.iteritems():
		print words[verbi]
		for typ, content in prop.iteritems():
			print typ, ": ", ' '.join(words[content[0]:content[1]+1])
	print ""

def printEval(fn, fp, tp, s):
	print "\n", s
	print "fn:", fn, "fp:", fp, "tp:", tp
	if tp+fn==0:
		recall = 0
	else:
		recall = tp/(tp+fn)
	if tp+fp==0:
		precision = 0
	else:
		precision = tp/(tp+fp)
	if recall+precision==0:
		fmeasure = 0
	else:
		fmeasure = 2*recall*precision/(recall+precision)
	print "recall", recall
	print "precision", precision
	print "fmeasure", fmeasure

def printCandidates(cds, words):
	for cd in cds:
		print ' '.join(words[cd.start:cd.end+1])

def bfs(root, sent, ans):
	q = deque()
	q.append(root)
	fn = 0.0
	fp = 0.0
	tp = 0.0
	print "\nsentence:"
	for word in sent:
		print word,
	# for i, word in enumerate(sent):
	# 	print i, word
	print ""
	while q:
		nd = q.popleft()
		if nd.type in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
			# print sent[nd.start]
			tfn, tfp, ttp = solve(nd, sent, ans)
			fn += tfn
			fp += tfp
			tp += ttp
		for child in nd.children:
			q.append(child)
	return (fn, fp, tp)

def bfs2(root, words, ans):
	q = deque()
	q.append(root)
	l = []
	while q:
		# print nd
		nd = q.popleft()
		if nd.type in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
			l.append(nd)
		for child in nd.children:
			q.append(child)
	l = sorted(l, key=lambda v: v.start)
	print "current sent:",
	for word in words:
		print word,
	print ""
	for n in l:
		solve(n, words, ans)

def solve(nd, sent, ans):
	cds = prune(nd)
	print "\n>Analyzing:", sent[nd.start]
	printCandidates(cds, sent)
	return evaluatePrune(cds, nd.start, ans, sent)

def differInPunc(a1,a2, b1, b2, sent):
	punc = (".", ":", "-LRB-", "-RRB-", ",", "--", "''", "``")
	if ((a1-b1==1 and sent[b1] in punc) or (b1-a1==1 and sent[a1] in punc) or a1==b1 ) and \
	((a2-b2==1 and sent[a2] in punc) or (b2-a2==1 and sent[b2] in punc) or a2==b2):
		return True
	else:
		return False

def evaluatePrune(cands, verbi, ans, sent):
	fn = 0.0
	fp = 0.0
	tp = 0.0
	if verbi in ans:
		for k,v in ans[verbi].iteritems():
			find = False
			for c in cands:
				if (c.start==v[0] and c.end==v[1]) or differInPunc(c.start,c.end,v[0],v[1],sent):
					find = True
					break
			if not find:
				fn+=1
		for c in cands:
			find = False
			for k,v in ans[verbi].iteritems():
				if (c.start==v[0] and c.end==v[1]) or differInPunc(c.start,c.end,v[0],v[1],sent):
					find = True
					tp+=1
					break
			if not find:
				fp+=1
		print "fn", fn, "fp", fp, "tp", tp
	return (fn, fp, tp)

def getParenth2(labels):
	pos = {}
	stack = []
	for i, l in enumerate(labels):
		if l[0]=="(":
			if ')' in l:
				pos[i] = i
			else:
				stack.append(i)
		elif ')' in l:
			pos[stack.pop()] = i
	return pos

def getVerb(labels):
	for i, l in enumerate(labels):
		if l[:3]=="(V*":
			return i

def getPropAns(sent):
	props = {}
	if len(sent)!=0:
		l = len(sent[0]) - 6
		if l>0:
			for i in xrange(l):
				i = i+6
				labels = []
				for word in sent:
					labels.append(word[i])
				parenth = getParenth2(labels)
				verbi = getVerb(labels)
				props[verbi] = {}
				for k,v in parenth.iteritems():
					if labels[k][:3]!="(V*":
						if k==v:
							props[verbi][labels[k][1:-2]] = (k,v)
						else:
							props[verbi][labels[k][1:-1]] = (k,v)
	return props

class SRL:
	def __init__(self, iter):
		self.iter = iter
		LEFT = 1
		RIGHT = -1
		self.headTable = {
			"ADJP": {
				"direction": LEFT,
				"priority_list": ("NNS", "QP", "NN", "$", "ADVP", "JJ", "VBN", "VBG", "ADJP", "JJR", "NP", "JJS", "DT", "FW", "RBR", "RBS", "SBAR", "RB")				
			},
			"ADVP": {
				"direction": RIGHT,
				"priority_list": ("RB", "RBR", "RBS", "FW", "ADVP", "TO", "CD", "JJR", "JJ", "IN", "NP", "JJS", "NN")
			},
			"CONJP": {
				"direction": RIGHT,
				"priority_list": ("CC", "RB", "IN")
			},
			"FRAG": {
				"direction": RIGHT,
				"priority_list": ()
			},
			"INTJ": {
				"direction": LEFT,
				"priority_list": ()
			},
			"LST": {
				"direction": RIGHT,
				"priority_list": ("LS", ":")
			},
			"NAC": {
				"direction": LEFT,
				"priority_list": ("NN", "NNS", "NNP", "NNPS", "NP", "NAC", "EX", "$", "CD", "QP", "PRP", "VBG", "JJ", "JJS", "JJR", "ADJP", "FW")
			},
			"PP": {
				"direction": LEFT,
				"priority_list": ("IN" ,"TO" ,"VBG" ,"VBN" ,"RP" ,"FW")
			},
			"PRN": {
				"direction": LEFT,
				"priority_list": ()
			},
			"PRT": {
				"direction": RIGHT,
				"priority_list": ("RP")
			},
			"QP": {
				"direction": LEFT,
				"priority_list": ("$", "IN", "NNS", "NN", "JJ", "RB", "DT", "CD", "NCD", "QP", "JJR", "JJS")
			},
			"RRC": {
				"direction": RIGHT,
				"priority_list": ("VP", "NP", "ADVP", "ADJP", "PP")
			},
			"S": {
				"direction": LEFT,
				"priority_list": ("TO", "IN", "VP", "S", "SBAR", "ADJP", "UCP", "NP")
			},
			"SBAR": {
				"direction": LEFT,
				"priority_list": ("WHNP", "WHPP", "WHADVP", "WHADJP", "IN", "DT", "S", "SQ", "SINV", "SBAR", "FRAG")
			},
			"SBARQ": {
				"direction": LEFT,
				"priority_list": ("SQ", "S", "SINV", "SBARQ", "FRAG")
			},
			"SINV": {
				"direction": LEFT,
				"priority_list": ("VBZ", "VBD", "VBP", "VB", "MD", "VP", "S", "SINV", "ADJP", "NP")
			},
			"SQ": {
				"direction": LEFT,
				"priority_list": ("VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ")
			},
			"UCP": {
				"direction": RIGHT,
				"priority_list": ()
			},
			"VP": {
				"direction": LEFT,
				"priority_list": ("TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG", "VBP", "VP", "ADJP", "NN", "NNS", "NP")
			},
			"WHADJP": {
				"direction": LEFT,
				"priority_list": ("CC", "WRB", "JJ", "ADJP")
			},
			"WHADVP": {
				"direction": RIGHT,
				"priority_list": ("CC", "WRB")
			},
			"WHNP": {
				"direction": LEFT,
				"priority_list": ("WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP")
			},
			"WHPP": {
				"direction": RIGHT,
				"priority_list": ("IN", "TO", "FW")
			}
		}

	def trainIdentifier(self, trainfile):
		self.read(trainfile)
		self.createFeatureIdx()
		self.createFeatureMat()
		self.X = self.X.tocsr()
		print "create identifier"
		self.identifier = SGDClassifier(loss="hinge", penalty="l2", n_iter=self.iter)
		print "fit feature matrix......"
		self.identifier.fit(self.X, self.y)
		print "save identifier and features to disk..."
		pickle.dump(self.featurei, open("ifeatures.map", "wb"))
		pickle.dump(self.identifier, open("identifier", "wb"))

	def trainClassifier(self):
		print "start training classifier..."
		self.numinst = 0
		for i in self.y:
			if i==1:
				self.numinst+=1
		self.yy = self.y
		self.createFeatureMat(self.notCandidate)
		print "create classifier"
		self.classifier = SGDClassifier(loss="hinge", penalty="l2", n_iter=self.iter)
		print "fit feature matrix......"
		self.classifier.fit(self.X, self.yl)
		print "save classifier to disk..."
		pickle.dump(self.classifier, open("classifier", "wb"))

	def getPropAns(self, si):
		props = {}
		sent = self.sentences[si]
		if len(sent)!=0:
			l = len(sent[0]) - 6
			if l>0:
				for i in xrange(l):
					i = i+6
					labels = []
					for word in sent:
						labels.append(word[i])
					parenth = getParenth2(labels)
					verbi = getVerb(labels)
					props[verbi] = {}
					for k,v in parenth.iteritems():
						if labels[k][:3]!="(V*":
							if k==v:
								props[verbi][(k,v)] = labels[k][1:-2]
							else:
								props[verbi][(k,v)] = labels[k][1:-1]
		return props

	def getHeadNPinPP(self, node):	
		for child in node.children:
			if child.type=="NP":
				return child
		for child in node.children:
			if child.type=="PP":
				return self.getHeadNPinPP(child)
		return None

	def findFeatures(self):
		print "findFeatures..."
		self.numinst = 0
		lmtzr = WordNetLemmatizer()
		for si, sent in enumerate(self.candidates):
			# print "find features in", si
			self.paths.append({})
			self.lemmas.append({})
			self.subcats.append({})
			self.heads.append({})
			self.parentHeads.append({})
			self.leftHeads.append({})
			self.rightHeads.append({})
			self.pathlens.append({})
			self.partialPaths.append({})
			ans = self.getPropAns(si)
			for verbi, candidates in sent.iteritems():
				if verbi in ans:
					self.paths[si][verbi] = []
					self.heads[si][verbi] = []
					self.parentHeads[si][verbi] = []
					self.leftHeads[si][verbi] = []
					self.rightHeads[si][verbi] = []
					self.pathlens[si][verbi] = []
					self.partialPaths[si][verbi] = []
					#path from verbi to root
					vtos = [self.getTag(verbi, si)]
					vnode = self.getNode(verbi, si)
					vp = vnode.parent
					while vp!=None:
						vtos.append(vp.type)
						vp = vp.parent
					#lemma of predicate
					lemma = lmtzr.lemmatize(self.getWord(verbi, si).lower(), 'v')
					self.lemmas[si][verbi] = lemma
					for candidate in candidates:
						# print "candidate:", ' '.join(self.getWords(si)[candidate.start:candidate.end+1])
						self.numinst+=1
						#find head word
						head = self.findHead(candidate, si)
						# print head
						self.heads[si][verbi].append(head)
						#parent head
						if candidate.parent!=None:
							self.parentHeads[si][verbi].append(self.findHead(candidate.parent, si))
						else:
							self.parentHeads[si][verbi].append(None)
						leftSibling = candidate.leftSibling()
						if leftSibling!=None:
							self.leftHeads[si][verbi].append(self.findHead(leftSibling, si))
						else:
							self.leftHeads[si][verbi].append(None)
						rightSibling = candidate.rightSibling()
						if rightSibling!=None:
							self.rightHeads[si][verbi].append(self.findHead(rightSibling, si))
						else:
							self.rightHeads[si][verbi].append(None)
						#PP NP Head-Word/Head-POS For a PP, retrieve the head-word /head-POS of its NP
						if candidate.type=="PP":
							candidate.headNP = self.getHeadNPinPP(candidate)
							if candidate.headNP!=None:
								candidate.headNP = self.findHead(candidate.headNP, si)
						#path
						path, pathlen, partialPath = self.findPath(vtos, candidate)
						self.paths[si][verbi].append(path)
						self.pathlens[si][verbi].append(pathlen)
						self.partialPaths[si][verbi].append(partialPath)
					#subcategorization
					if vnode.parent!=None:
						subcategorization = ' '.join([child.type for child in vnode.parent.children])
						self.subcats[si][verbi] = subcategorization

	def initFeatureCache(self):
		self.subcats = []
		self.paths = []
		self.lemmas = []
		self.heads = []
		self.parentHeads = []
		self.leftHeads = []
		self.rightHeads = []
		self.pathlens = []
		self.partialPaths = []

	def createFeatureIdx(self):
		self.numinst = 0
		self.features=[]
		self.fcount = 0
		self.featurei={}
		self.curSentIdx = 0
		numsent = 0
		self.initFeatureCache()
		print "create identify features index..."
		feature_list = ("passive_voice", "active_voice", "predicate_tag_VB", "predicate_tag_VBD", 
			"predicate_tag_VBG", "predicate_tag_VBN", "predicate_tag_VBP", "predicate_tag_VBZ", 
			"pos_before", "pos_after", "passive_voice_pos_before", "active_voice_pos_before", "active_voice_pos_after", "passive_voice_pos_after")
		for feature in feature_list:
			self.addFeature(feature)
		self.findFeatures()
		for si, sent in enumerate(self.candidates):
			# print "create features row for", si
			ans = self.getPropAns(si)
			numsent+=1
			for verbi, candidates in sent.iteritems():
				if verbi in ans:
					#lemma of predicate
					self.addFeature("predicate_word_"+self.lemmas[si][verbi])
					for ci, candidate in enumerate(candidates):
						self.numinst+=1
						#phrase type of candidate
						self.addFeature("candidate_tag_"+candidate.type)
						#lemma & phrase type
						self.addFeature("predicate_word_"+self.lemmas[si][verbi]+"_phrase_type_"+candidate.type)
						#find head words
						self.addFeature("candidate_head_tag_"+self.heads[si][verbi][ci]["tag"])
						self.addFeature("candidate_head_word_"+self.heads[si][verbi][ci]["word"].lower())
						#path
						self.addFeature("path_"+self.paths[si][verbi][ci])
						#predicate word & path
						self.addFeature("predicate_word_"+self.lemmas[si][verbi]+"_path_"+self.paths[si][verbi][ci])
						#path length
						self.addFeature("pathlen_"+str(self.pathlens[si][verbi][ci]))
						#partial path
						self.addFeature("partialPath_"+self.partialPaths[si][verbi][ci])
						#first word last word and pos
						self.addFeature("first_word_"+self.getWord(candidate.start, si).lower())
						self.addFeature("last_word_"+self.getWord(candidate.end, si).lower())
						self.addFeature("first_word_tag_"+self.getTag(candidate.start, si))
						self.addFeature("last_word_tag_"+self.getTag(candidate.end, si))
						#two words before and after the argument and pos
						if candidate.start>0:
							self.addFeature("first_word_before_"+self.getWord(candidate.start-1, si).lower())
							self.addFeature("first_word_before_tag_"+self.getTag(candidate.start-1, si))
						if candidate.start>1:
							self.addFeature("second_word_before_"+self.getWord(candidate.start-2, si).lower())
							self.addFeature("second_word_before_tag_"+self.getTag(candidate.start-2, si))
						if candidate.end<len(self.sentences[si])-1:
							self.addFeature("first_word_after_"+self.getWord(candidate.end+1, si).lower())
							self.addFeature("first_word_after_tag_"+self.getTag(candidate.end+1, si))
						if candidate.end<len(self.sentences[si])-2:
							self.addFeature("second_word_after_"+self.getWord(candidate.end+2, si).lower())
							self.addFeature("second_word_after_tag_"+self.getTag(candidate.end+2, si))
						#predicate lemma & head word
						self.addFeature("predicate_word_" + self.lemmas[si][verbi]+"_head_word_"+self.heads[si][verbi][ci]["word"].lower())
						#parent head and head pos
						if candidate.parent!=None and self.parentHeads[si][verbi][ci]!=None:
							self.addFeature("parent_head_tag_"+self.parentHeads[si][verbi][ci]["tag"])
							self.addFeature("parent_head_word_"+self.parentHeads[si][verbi][ci]["word"].lower())
						#parent phrase type
						if candidate.parent!=None:
							self.addFeature("parent_tag_"+candidate.parent.type)
						#left and right sibling phrase type, head word, and head POS tag
						if self.leftHeads[si][verbi][ci]!=None:
							self.addFeature("left_head_tag_"+self.leftHeads[si][verbi][ci]["tag"])
							self.addFeature("left_head_word_"+self.leftHeads[si][verbi][ci]["word"].lower())
						leftSibling = candidate.leftSibling()
						if leftSibling!=None:
							self.addFeature("left_tag_"+leftSibling.type)
						if self.rightHeads[si][verbi][ci]!=None:
							self.addFeature("right_head_tag_"+self.rightHeads[si][verbi][ci]["tag"])
							self.addFeature("right_head_word_"+self.rightHeads[si][verbi][ci]["word"].lower())
						rightSibling = candidate.rightSibling()
						if rightSibling!=None:
							self.addFeature("right_tag_"+rightSibling.type)
						#head of pp parent
						if candidate.parent!=None and candidate.parent.type=="PP" and self.parentHeads[si][verbi][ci]!=None:
							self.addFeature("headofPP_"+self.parentHeads[si][verbi][ci]["word"].lower())
							#lemma & head of pp parent
							self.addFeature("predicate_word_"+self.lemmas[si][verbi]+"_headofPPP_"+self.parentHeads[si][verbi][ci]["word"].lower())
						#PP NP Head-Word/Head-POS For a PP, retrieve the head-word /head-POS of its NP
						if candidate.headNP!=None:
							self.addFeature("PPNPhead_word_"+candidate.headNP["word"].lower())
							self.addFeature("PPNPhead_tag_"+candidate.headNP["tag"])
					#subcategorization
					if verbi in self.subcats[si]:
						self.addFeature("subcat_"+self.subcats[si][verbi])
		# print self.features
		print "number of features:", self.fcount
		print "number of sentences:", numsent
		print "number of instances:", self.numinst
		
	def createFeatureMat(self, filter=None):
		print "create feature matrix..."
		self.X = lil_matrix((self.numinst, self.fcount))
		self.y = [0 for i in xrange(self.numinst)]
		self.yl = [0 for i in xrange(self.numinst)]
		self.numinst = 0
		count = -1
		labels = ["NULL","A0","A1","A2","A3","A4","A5","AA","AM",
			"R-A0","R-A1", "R-A2", "R-A3","R-A4","R-AA",
			"AM-ADV","AM-CAU","AM-DIR","AM-DIS","AM-EXT","AM-LOC","AM-MNR",
			"AM-MOD","AM-NEG","AM-PNC","AM-PRD","AM-REC","AM-TMP",
			"R-AM-MNR","R-AM-LOC","R-AM-TMP","R-AM-PNC","R-AM-ADV","R-AM-CAU","R-AM-DIR","R-AM-EXT",
			"V","C-A0" , "C-A1", "C-A2", "C-A3", "C-A4","C-V", "C-AM-MNR", "C-AM-LOC", "C-AM-TMP", 
			"C-AM-EXT", "C-AM-NEG", "C-AM-ADV", "C-AM-DIS", "C-AM-CAU", "C-AM-DIR", "C-AM-PNC"]
		self.labels = labels
		ltoi = {}
		for i, l in enumerate(labels):
			ltoi[l] = i
		for si, sent in enumerate(self.candidates):
			# print "\nsentence:", ' '.join(self.getWords(si))
			ans = self.getPropAns(si)
			for verbi, candidates in sent.iteritems():
				if verbi in ans:
					# print "\nverb:", self.getWord(verbi, si)
					for ci, candidate in enumerate(candidates):
						count+=1
						if filter!=None and filter(count):
							continue
						# print "\ncandidate:", ' '.join(self.getWords(si)[candidate.start:candidate.end+1])
						# if candidate.parent!=None:
							# print "candidate parent:", ' '.join(self.getWords(si)[candidate.parent.start:candidate.parent.end+1])
							# print "candidate parent type:", candidate.parent.type
						#lemma of predicate
						# print "predicate lemma:", self.lemmas[si][verbi]
						self.setFeature("predicate_word_"+self.lemmas[si][verbi])
						#pos of predicate
						self.setFeature("predicate_tag_"+self.getTag(verbi, si))
						#voice
						# print "voice:", self.voice(verbi, si)
						self.setFeature(self.voice(verbi, si))
						#candidate type
						self.setFeature("candidate_tag_"+candidate.type)
						#lemma & phrase type
						self.setFeature("predicate_word_"+self.lemmas[si][verbi]+"_phrase_type_"+candidate.type)
						#head word
						# print "head word:", self.heads[si][verbi][ci]["word"]
						self.setFeature("candidate_head_word_"+self.heads[si][verbi][ci]["word"].lower())
						#pos of head word
						# print "head pos:", self.heads[si][verbi][ci]["tag"]
						self.setFeature("candidate_head_tag_"+self.heads[si][verbi][ci]["tag"])
						#position
						# print "position:", self.beforeOrAfter(verbi, candidate.start)
						self.setFeature(self.beforeOrAfter(verbi, candidate.start))
						#path
						# print "path:", self.paths[si][verbi][ci]
						self.setFeature("path_"+self.paths[si][verbi][ci])
						#predicate word & path
						self.setFeature("predicate_word_"+self.lemmas[si][verbi]+"_path_"+self.paths[si][verbi][ci])
						#path length
						# print "path length:", str(self.pathlens[si][verbi][ci])
						self.setFeature("pathlen_"+str(self.pathlens[si][verbi][ci]))
						#partial path
						# print "partial path:", self.partialPaths[si][verbi][ci]
						self.setFeature("partialPath_"+self.partialPaths[si][verbi][ci])
						#subcategorization
						# print "subcat:", self.subcats[si][verbi]
						self.setFeature("subcat_"+self.subcats[si][verbi])
						#first word last word
						self.setFeature("first_word_"+self.getWord(candidate.start, si).lower())
						self.setFeature("last_word_"+self.getWord(candidate.end, si).lower())
						self.setFeature("first_word_tag_"+self.getTag(candidate.start, si))
						self.setFeature("last_word_tag_"+self.getTag(candidate.end, si))
						#two words before and after the argument and pos
						if candidate.start>0:
							self.setFeature("first_word_before_"+self.getWord(candidate.start-1, si).lower())
							self.setFeature("first_word_before_tag_"+self.getTag(candidate.start-1, si))
						if candidate.start>1:
							self.setFeature("second_word_before_"+self.getWord(candidate.start-2, si).lower())
							self.setFeature("second_word_before_tag_"+self.getTag(candidate.start-2, si))
						if candidate.end<len(self.sentences[si])-1:
							self.setFeature("first_word_after_"+self.getWord(candidate.end+1, si).lower())
							self.setFeature("first_word_after_tag_"+self.getTag(candidate.end+1, si))
						if candidate.end<len(self.sentences[si])-2:
							self.setFeature("second_word_after_"+self.getWord(candidate.end+2, si).lower())
							self.setFeature("second_word_after_tag_"+self.getTag(candidate.end+2, si))
						#predicate lemma & head word
						self.setFeature("predicate_word_" + self.lemmas[si][verbi]+"_head_word_"+self.heads[si][verbi][ci]["word"].lower())
						#voice & position
						self.setFeature(self.voice(verbi, si)+"_"+self.beforeOrAfter(verbi, candidate.start))
						#parent head
						if candidate.parent!=None and self.parentHeads[si][verbi][ci]!=None:
							self.setFeature("parent_head_tag_"+self.parentHeads[si][verbi][ci]["tag"])
							self.setFeature("parent_head_word_"+self.parentHeads[si][verbi][ci]["word"].lower())
						#parent phrase type
						if candidate.parent!=None:
							self.setFeature("parent_tag_"+candidate.parent.type)
						#left and right sibling phrase type, head word, and head POS tag
						if self.leftHeads[si][verbi][ci]!=None:
							# print "left_head_tag_"+self.leftHeads[si][verbi][ci]["tag"]
							# print "left_head_word_"+self.leftHeads[si][verbi][ci]["word"].lower()
							self.setFeature("left_head_tag_"+self.leftHeads[si][verbi][ci]["tag"])
							self.setFeature("left_head_word_"+self.leftHeads[si][verbi][ci]["word"].lower())
						leftSibling = candidate.leftSibling()
						if leftSibling!=None:
							# print "left_tag_"+leftSibling.type
							self.setFeature("left_tag_"+leftSibling.type)
						if self.rightHeads[si][verbi][ci]!=None:
							# print "right_head_tag_"+self.rightHeads[si][verbi][ci]["tag"]
							# print "right_head_word_"+self.rightHeads[si][verbi][ci]["word"].lower()
							self.setFeature("right_head_tag_"+self.rightHeads[si][verbi][ci]["tag"])
							self.setFeature("right_head_word_"+self.rightHeads[si][verbi][ci]["word"].lower())
						rightSibling = candidate.rightSibling()
						if rightSibling!=None:
							# print "right_tag_"+rightSibling.type
							self.setFeature("right_tag_"+rightSibling.type)
						#head of pp parent
						if candidate.parent!=None and candidate.parent.type=="PP" and self.parentHeads[si][verbi][ci]!=None:
							# print "headofPP_"+self.parentHeads[si][verbi][ci]["word"].lower()
							self.setFeature("headofPP_"+self.parentHeads[si][verbi][ci]["word"].lower())
							#lemma & head of pp parent
							self.setFeature("predicate_word_"+self.lemmas[si][verbi]+"_headofPPP_"+self.parentHeads[si][verbi][ci]["word"].lower())
						#PP NP Head-Word/Head-POS For a PP, retrieve the head-word /head-POS of its NP
						if candidate.headNP!=None:
							# print "PPNPhead_word_"+candidate.headNP["word"].lower()
							self.setFeature("PPNPhead_word_"+candidate.headNP["word"].lower())
							self.setFeature("PPNPhead_tag_"+candidate.headNP["tag"])
						#check with ans. set y vector
						find = False
						for k, v in ans[verbi].iteritems():
							if (candidate.start==k[0] and candidate.end==k[1]) or differInPunc(candidate.start,candidate.end,k[0],k[1],self.getWords(si)):
								find = True
								# print "is candidate"
								self.y[self.numinst] = 1
								self.yl[self.numinst] = ltoi[v]
								break
						if not find:
							# print "not a candidate"
							self.y[self.numinst] = 0
						self.numinst += 1

	def printLabels(self):
		count = 0
		c = -1
		for si, sent in enumerate(self.candidates):
			print "\nsentence:", ' '.join(self.getWords(si))
			ans = self.getPropAns(si)
			for verbi, candidates in sent.iteritems():
				if verbi in ans:
					print "\nverb:", self.getWord(verbi, si)
					for ci, candidate in enumerate(candidates):
						c+=1
						if self.notCandidate(c):
							# print "not a candidate"
							continue
						print "\ncandidate:", ' '.join(self.getWords(si)[candidate.start:candidate.end+1]), 
						print "label:", self.labels[self.yyl[count]],
						print "ans:", self.labels[self.yl[count]]
						count+=1

	def findHead(self, nd, si):
		LEFT = 1
		RIGHT = -1
		headTable = self.headTable
		if nd.start==nd.end:
			return {
				"tag": nd.type,
				"word": self.getWord(nd.start, si)
			}
		elif nd.type in headTable:
			if len(headTable[nd.type]["priority_list"])!=0:
				for tag in headTable[nd.type]["priority_list"]:
					#find head phrase from children
					for child in nd.children[::headTable[nd.type]["direction"]]:
						if child.type==tag:
							return self.findHead(child, si)
					#find head word from leaves
					for wi in range(nd.start, nd.end+1)[::headTable[nd.type]["direction"]]:
						if self.getTag(wi, si)==tag:
							return  {
								"tag": tag,
								"word": self.getWord(wi, si)
							}
			if headTable[nd.type]["direction"]==LEFT:
				return {
					"tag": self.getTag(nd.start, si),
					"word": self.getWord(nd.start, si)
				}
			else:
				return {
					"tag": self.getTag(nd.end, si),
					"word": self.getWord(nd.end, si)
				}
		elif nd.type=="NP":
			# print "for:", self.getWords(si)[nd.start:nd.end+1]
			if self.getTag(nd.end, si)=="POS":
				return {
					"tag": "POS",
					"word": self.getWord(nd.end, si)
				}
			else:
				tags = {"NN", "NNP", "NNPS", "NNS", "NX", "POS"}
				# print "number of children:", len(nd.children)
				for child in nd.children:
					# print "has children:", child.type
					if child.type == "NP":
						# print "choose leftmost np:", self.getWords(si)[child.start:child.end+1]
						return self.findHead(child, si)
				for wi in range(nd.end, nd.start-1, -1):
					if self.getTag(wi, si) in tags:
						# print "choose rightmost noun:", self.getWord(wi, si)
						return {
							"tag": self.getTag(wi, si),
							"word": self.getWord(wi, si)
						}
				for child in nd.children[::-1]:
					if child.type=="NX":
						return self.findHead(child, si)
				tags = {"JJ", "JJS", "JJR", "RB", "$", "CD", "-RRB-"}
				for wi in range(nd.end, nd.start-1, -1):
					if self.getTag(wi, si) in tags:
						return {
							"tag": self.getTag(wi, si),
							"word": self.getWord(wi, si)
						}
				return {
					"tag": self.getTag(nd.end, si),
					"word": self.getWord(nd.end, si)
				}
		else:
			# print "unknown node type in findHead:", nd.type
			return {
				"tag": nd.type,
				"word": self.getWord(nd.end, si)
			}

	def loadFeatures(self, ffile):
		print "load features"
		self.featurei = pickle.load(open(ffile, "rb"))

	def loadIdentidier(self, idfile):
		print "load identifier"
		self.identifier = pickle.load(open(idfile, "rb"))

	def loadClassifier(self, clffile):
		print "load classifier"
		tar = tarfile.open(clffile)
		self.classifier = pickle.load(tar.extractfile("classifier"))

	def read(self, f):
		self.parseTrees = []
		self.sentences = []
		self.candidates = []
		print "open files..."
		data = open(f)
		pf = open(f+".reparse")
		print "read sentences..."
		for sentence in self.fileIter(data):
			self.sentences.append(sentence)
			self.candidates.append({})
		print "contruct parse trees..."
		for line in pf:
			if len(line)<=1:
				break
			self.parseTrees.append(construct2(line))
		print "update tags..."
		for i, parse_tree in enumerate(self.parseTrees):
			self.updateTag(i, parse_tree)
		print "prune..."
		for i, root in enumerate(self.parseTrees):
			self.curSentIdx = i
			self.bfs(root, self.getCandidates)

	def identify(self, f):
		self.initFeatureCache()
		self.fcount = len(self.featurei)
		self.read(f)
		self.findFeatures()
		self.createFeatureMat()
		# self.yy = [0 for i in xrange(self.numinst)]
		print "numinst:", self.numinst
		# print "nrow in x:", len(self.X)
		# for i, x in enumerate(self.X):
		# for i in xrange(self.numinst):
		# 	print "predict", i
		# 	self.yy[i] = self.identifier.predict(self.X[i,:])
		print "start identification..."
		self.X = self.X.tocsr()
		self.yy = self.identifier.predict(self.X)

	def notCandidate(self, ninst):
		if self.yy[ninst]==0:
			return True
		else:
			return False

	def classify(self):
		#probably need to init feature array
		#add classification features if needed
		# num = self.yy.count(1)
		self.numinst = 0
		for i in self.yy:
			if i==1:
				self.numinst+=1
		self.createFeatureMat(self.notCandidate)
		# yl = []
		# self.yyl = []
		# j = 0
		# for i, l in enumerate(self.yy):
			# print "identified label:",l
			# if l==1:
				# X.append(self.X[i])
				# X[j, :] = self.X[i, :]
				# yl.append(self.yl[i])
				# self.yyl.append(0)
		# X = X.tocsr()
		# self.yl = yl
		# for i in xrange(num):
			# print "predict", i
			# self.yyl[i] = self.classifier.predict(X[i,:])
		self.X = self.X.tocsr()
		print "start classification..."
		self.yyl = self.classifier.predict(self.X)

	def evalIdentifier(self):
		fn = 0.
		fp = 0.
		tp = 0.
		for i, label in enumerate(self.yy):
			if label==self.y[i]:
				tp+=1
			elif self.y[i]==1:
				fn+=1
			else:
				fp+=1
		print fn, fp, tp
		printEval(fn, fp, tp, "identifier evaluation:")

	def evalClassifier(self):
		confusion = [[0. for i in xrange(len(self.labels))] for j in xrange(len(self.labels))]
		for i, label in enumerate(self.yyl):
			if label==self.yl[i]:
				confusion[label][label] += 1
			else:
				confusion[self.yl[i]][label] += 1
		colsum =  [sum(x) for x in zip(*confusion)]
		for expect, data in enumerate(confusion):
			
			tp = data[expect]
			fn = sum(data) - tp
			fp = colsum[expect] - tp
			if abs(tp)<1e-8 and abs(fn)<1e-8 and abs(fp)<1e-8:
				continue
			if tp+fp+fn<2:
				continue
			printEval(fn, fp, tp, self.labels[expect]+" evaluation:")
			for actual, count in enumerate(data):
				if abs(count) < 1e-8:
					continue
				print "expect:", self.labels[expect], "actual:", self.labels[actual], count

	def getWords(self, si):
		return [word[0] for word in self.sentences[si]]

	def findPath(self, vtos, candidate):
		ctos = [candidate.type]
		cp = candidate.parent
		while cp!=None:
			ctos.append(cp.type)
			cp = cp.parent
		pi = -1
		vtosl = len(vtos)
		ctosl = len(ctos)
		while -pi+1 <= vtosl and -pi+1 <= ctosl and vtos[pi-1]==ctos[pi-1]:
			pi-=1
		return ("+".join(vtos[:vtosl+pi+1]) + "-" + "-".join(ctos[:ctosl+pi][::-1]), 
			vtosl+ctosl+pi*2-1, 
			"-".join(ctos[:ctosl+pi+1]))

	def beforeOrAfter(self, vi, ci):
		if ci < vi:
			return "pos_before"
		else:
			return "pos_after"

	def voice(self, vi, si):
		ret = "active_voice"
		inNP = False
		vb = self.getNode(vi, si)
		if vb.type=="VBN":
			vp = vb.parent
			if vp==None:
				return "passive_voice"
			lastVP = None
			while vp!=None and vp.type!="S":
				if vp.type=="NP":
					inNP = True
					break
				if vp.type=="VP":
					lastVP=vp
				vp = vp.parent
			if inNP:
				ret = "passive_voice"
			else:
				if lastVP!=None:
					have = {"has", "have", "had", "having", "'ve", "'d"}
					be = {"is", "am", "are", "was", "were", "been", "be", "being", "'s", "'re", "ai"}
					for i in xrange(vb.start-1, lastVP.start-1, -1):
						word = self.getWord(i, si)
						if word in be:
							ret = "passive_voice"
							break
						elif word in have:
							break
				else:
					ret = "passive_voice"
		return ret

	def setFeature(self, feature):
		i = self.numinst
		if feature in self.featurei:
			self.X[i, self.featurei[feature]] = 1
		# else:
			# print "not add before", feature

	def addFeature(self, feature):
		if feature not in self.featurei:
			self.features.append(feature)
			self.featurei[feature] = self.fcount
			self.fcount+=1

	def updateTag(self, si, root):
		q = deque()
		q.append(root)
		while q:
			nd = q.popleft()
			if nd.start==nd.end:
				self.setTag(nd.start, si, nd.type)
			for child in nd.children:
				q.append(child)

	def getNode(self, ni, si):
		q = deque()
		q.append(self.parseTrees[si])
		while q:
			nd = q.popleft()
			if nd.start==nd.end and nd.start == ni:
				return nd
			for child in nd.children:
				q.append(child)
		return None

	def getWord(self, wi, si):
		return self.sentences[si][wi][0]

	def getTag(self, wi, si):
		return self.sentences[si][wi][1]

	def setTag(self, wi, si, tag):
		self.sentences[si][wi][1] = tag

	def getCandidates(self, nd):
		pruneResult = prune(nd)
		# print "\n>Analyzing:", self.getWord(nd.start, self.curSentIdx)
		# printCandidates(pruneResult, self.getWords(self.curSentIdx))
		self.candidates[self.curSentIdx][nd.start] = pruneResult

	def bfs(self, root, func):
		vbs = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
		q = deque()
		q.append(root)
		while q:
			nd=q.popleft()
			if nd.type in vbs:
				func(nd)
			for child in nd.children:
				q.append(child)

	def fileIter(self, data):
		sentence = []
		for line in data:
			if len(line.split())==0:
				if len(sentence)==0:
					continue
				yield sentence
				sentence = []
			else:
				parts = line.split()
				sentence.append(parts)

def main():
	srl = SRL(40)
	# srl.trainIdentifier("10000lines")
	# srl.trainClassifier()
	srl.loadFeatures("features40")
	srl.loadIdentidier("identifier40")
	srl.loadClassifier("clf40.tar.gz")
	srl.identify("wsjtest")
	srl.evalIdentifier()
	srl.classify()
	srl.evalClassifier()
	# srl.printLabels()

if __name__ == "__main__":
	main()