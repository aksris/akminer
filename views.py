#Hello Changes
from flask import *
from werkzeug import *
from jinja2 import *
import os
import csv
import sys
import json
import random
import math
import operator

from itertools import chain, combinations, imap
from collections import defaultdict, namedtuple
from optparse import OptionParser

UPLOAD_FOLDER = '/home/ak/flaskr/uploads'
ALLOWED_EXTENSIONS = set(['txt','xls','xslx','csv'])

app = Flask(__name__)
app.secret_key = 'akminer'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 *1024 * 1024

super_f = {}

'''k nearest neighbors'''
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)-1):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])
 
 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

'''fp growth'''
from collections import defaultdict, namedtuple
from itertools import imap


def find_frequent_itemsets(transactions, minimum_support, include_support=False):
	items = defaultdict(lambda: 0) # mapping from items to their supports
	processed_transactions = []

	# Load the passed-in transactions and count the support that individual
	# items have.
	for transaction in transactions:
		processed = []
		for item in transaction:
			items[item] += 1
			processed.append(item)
		processed_transactions.append(processed)

	# Remove infrequent items from the item support dictionary.
	items = dict((item, support) for item, support in items.iteritems()
		if support >= minimum_support)

	def clean_transaction(transaction):
		transaction = filter(lambda v: v in items, transaction)
		transaction.sort(key=lambda v: items[v], reverse=True)
		return transaction

	master = FPTree()
	for transaction in imap(clean_transaction, processed_transactions):
		master.add(transaction)

	def find_with_suffix(tree, suffix):
		for item, nodes in tree.items():
			support = sum(n.count for n in nodes)
			if support >= minimum_support and item not in suffix:
				# New winner!
				found_set = [item] + suffix
				yield (found_set, support) if include_support else found_set

				# Build a conditional tree and recursively search for frequent
				# itemsets within it.
				cond_tree = conditional_tree_from_paths(tree.prefix_paths(item),
					minimum_support)
				for s in find_with_suffix(cond_tree, found_set):
					yield s # pass along the good news to our caller

	# Search for frequent itemsets, and yield the results we find.
	for itemset in find_with_suffix(master, []):
		yield itemset

class FPTree(object):
	"""
	An FP tree.

	This object may only store transaction items that are hashable (i.e., all
	items must be valid as dictionary keys or set members).
	"""

	Route = namedtuple('Route', 'head tail')

	def __init__(self):
		# The root node of the tree.
		self._root = FPNode(self, None, None)
		self._routes = {}

	@property
	def root(self):
		"""The root node of the tree."""
		return self._root

	def add(self, transaction):
		"""
		Adds a transaction to the tree.
		"""

		point = self._root

		for item in transaction:
			next_point = point.search(item)
			if next_point:
				next_point.increment()
			else:
				next_point = FPNode(self, item)
				point.add(next_point)

				self._update_route(next_point)

			point = next_point

	def _update_route(self, point):
		"""Add the given node to the route through all nodes for its item."""
		assert self is point.tree

		try:
			route = self._routes[point.item]
			route[1].neighbor = point # route[1] is the tail
			self._routes[point.item] = self.Route(route[0], point)
		except KeyError:
			# First node for this item; start a new route.
			self._routes[point.item] = self.Route(point, point)

	def items(self):
		"""
		Generate one 2-tuples for each item represented in the tree. The first
		element of the tuple is the item itself, and the second element is a
		generator that will yield the nodes in the tree that belong to the item.
		"""
		for item in self._routes.iterkeys():
			yield (item, self.nodes(item))

	def nodes(self, item):
		"""
		Generates the sequence of nodes that contain the given item.
		"""

		try:
			node = self._routes[item][0]
		except KeyError:
			return

		while node:
			yield node
			node = node.neighbor

	def prefix_paths(self, item):
		"""Generates the prefix paths that end with the given item."""

		def collect_path(node):
			path = []
			while node and not node.root:
				path.append(node)
				node = node.parent
			path.reverse()
			return path

		return (collect_path(node) for node in self.nodes(item))

	def inspect(self):
		print 'Tree:'
		self.root.inspect(1)

		print
		print 'Routes:'
		for item, nodes in self.items():
			print '  %r' % item
			for node in nodes:
				print '    %r' % node
		print '\n'

	def _removed(self, node):
		"""Called when `node` is removed from the tree; performs cleanup."""

		head, tail = self._routes[node.item]
		if node is head:
			if node is tail or not node.neighbor:
				# It was the sole node.
				del self._routes[node.item]
			else:
				self._routes[node.item] = self.Route(node.neighbor, tail)
		else:
			for n in self.nodes(node.item):
				if n.neighbor is node:
					n.neighbor = node.neighbor # skip over
					if node is tail:
						self._routes[node.item] = self.Route(head, n)
					break

def conditional_tree_from_paths(paths, minimum_support):
	"""Builds a conditional FP-tree from the given prefix paths."""
	tree = FPTree()
	condition_item = None
	items = set()

	# Import the nodes in the paths into the new tree. Only the counts of the
	# leaf notes matter; the remaining counts will be reconstructed from the
	# leaf counts.
	for path in paths:
		if condition_item is None:
			condition_item = path[-1].item

		point = tree.root
		for node in path:
			next_point = point.search(node.item)
			if not next_point:
				# Add a new node to the tree.
				items.add(node.item)
				count = node.count if node.item == condition_item else 0
				next_point = FPNode(tree, node.item, count)
				point.add(next_point)
				tree._update_route(next_point)
			point = next_point

	assert condition_item is not None

	# Calculate the counts of the non-leaf nodes.
	for path in tree.prefix_paths(condition_item):
		count = path[-1].count
		for node in reversed(path[:-1]):
			node._count += count

	# Eliminate the nodes for any items that are no longer frequent.
	for item in items:
		support = sum(n.count for n in tree.nodes(item))
		if support < minimum_support:
			# Doesn't make the cut anymore
			for node in tree.nodes(item):
				if node.parent is not None:
					node.parent.remove(node)

	# Finally, remove the nodes corresponding to the item for which this
	# conditional tree was generated.
	for node in tree.nodes(condition_item):
		if node.parent is not None: # the node might already be an orphan
			node.parent.remove(node)

	return tree

class FPNode(object):
	"""A node in an FP tree."""

	def __init__(self, tree, item, count=1):
		self._tree = tree
		self._item = item
		self._count = count
		self._parent = None
		self._children = {}
		self._neighbor = None

	def add(self, child):
		"""Adds the given FPNode `child` as a child of this node."""

		if not isinstance(child, FPNode):
			raise TypeError("Can only add other FPNodes as children")

		if not child.item in self._children:
			self._children[child.item] = child
			child.parent = self

	def search(self, item):
		"""
		Checks to see if this node contains a child node for the given item.
		If so, that node is returned; otherwise, `None` is returned.
		"""

		try:
			return self._children[item]
		except KeyError:
			return None

	def remove(self, child):
		try:
			if self._children[child.item] is child:
				del self._children[child.item]
				child.parent = None
				self._tree._removed(child)
				for sub_child in child.children:
					try:
						# Merger case: we already have a child for that item, so
						# add the sub-child's count to our child's count.
						self._children[sub_child.item]._count += sub_child.count
						sub_child.parent = None # it's an orphan now
					except KeyError:
						# Turns out we don't actually have a child, so just add
						# the sub-child as our own child.
						self.add(sub_child)
				child._children = {}
			else:
				raise ValueError("that node is not a child of this node")
		except KeyError:
			raise ValueError("that node is not a child of this node")

	def __contains__(self, item):
		return item in self._children

	@property
	def tree(self):
		"""The tree in which this node appears."""
		return self._tree

	@property
	def item(self):
		"""The item contained in this node."""
		return self._item

	@property
	def count(self):
		"""The count associated with this node's item."""
		return self._count

	def increment(self):
		"""Increments the count associated with this node's item."""
		if self._count is None:
			raise ValueError("Root nodes have no associated count.")
		self._count += 1

	@property
	def root(self):
		"""True if this node is the root of a tree; false if otherwise."""
		return self._item is None and self._count is None

	@property
	def leaf(self):
		"""True if this node is a leaf in the tree; false if otherwise."""
		return len(self._children) == 0

	def parent():
		doc = "The node's parent."
		def fget(self):
			return self._parent
		def fset(self, value):
			if value is not None and not isinstance(value, FPNode):
				raise TypeError("A node must have an FPNode as a parent.")
			if value and value.tree is not self.tree:
				raise ValueError("Cannot have a parent from another tree.")
			self._parent = value
		return locals()
	parent = property(**parent())

	def neighbor():
		doc = """
		The node's neighbor; the one with the same value that is "to the right"
		of it in the tree.
		"""
		def fget(self):
			return self._neighbor
		def fset(self, value):
			if value is not None and not isinstance(value, FPNode):
				raise TypeError("A node must have an FPNode as a neighbor.")
			if value and value.tree is not self.tree:
				raise ValueError("Cannot have a neighbor from another tree.")
			self._neighbor = value
		return locals()
	neighbor = property(**neighbor())

	@property
	def children(self):
		"""The nodes that are children of this node."""
		return tuple(self._children.itervalues())

	def inspect(self, depth=0):
		print ('  ' * depth) + repr(self)
		for child in self.children:
			child.inspect(depth + 1)

	def __repr__(self):
		if self.root:
			return "<%s (root)>" % type(self).__name__
		return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)



'''apriori'''
def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

def subsets(arr):
	""" Returns non empty subsets of arr"""
	return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
	"""calculates the support for items in the itemSet and returns a subset
 of the itemSet each of whose elements satisfies the minimum support"""
	_itemSet = set()
	localSet = defaultdict(int)

	for item in itemSet:
		for transaction in transactionList:
			if item.issubset(transaction):
				freqSet[item] += 1
				localSet[item] += 1

	for item, count in localSet.items():
		support = float(count)/len(transactionList)
		if support >= minSupport:
			_itemSet.add(item)
	return _itemSet


def joinSet(itemSet, length):
	"""Join a set with itself and returns the n-element itemsets"""
	return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
	transactionList = list()
	itemSet = set()
	for record in data_iterator:
		transaction = frozenset(record)
		transactionList.append(transaction)
		for item in transaction:
			itemSet.add(frozenset([item]))              # Generate 1-itemSets
	return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
	"""
	run the apriori algorithm. data_iter is a record iterator
	Return both:
	 - items (tuple, support)
	 - rules ((pretuple, posttuple), confidence)
	"""
	itemSet, transactionList = getItemSetTransactionList(data_iter)

	freqSet = defaultdict(int)
	largeSet = dict()
	# Global dictionary which stores (key=n-itemSets,value=support)
	# which satisfy minSupport

	assocRules = dict()
	# Dictionary which stores Association Rules

	oneCSet = returnItemsWithMinSupport(itemSet,
																			transactionList,
																			minSupport,
																			freqSet)

	currentLSet = oneCSet
	k = 2
	while(currentLSet != set([])):
		largeSet[k-1] = currentLSet
		currentLSet = joinSet(currentLSet, k)
		currentCSet = returnItemsWithMinSupport(currentLSet,
																						transactionList,
																						minSupport,
																						freqSet)
		currentLSet = currentCSet
		k = k + 1

	def getSupport(item):
		"""local function which Returns the support of an item"""
		return float(freqSet[item])/len(transactionList)

	toRetItems = []
	for key, value in largeSet.items():
		toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

	toRetRules = []
	count = 0
	for key, value in largeSet.items()[1:]:
		for item in value:
			_subsets = map(frozenset, [x for x in subsets(item)])
			for element in _subsets:
					remain = item.difference(element)
					if len(remain) > 0:
						confidence = getSupport(item)/getSupport(element)
						if confidence >= minConfidence:
							toRetRules.append(((tuple(element), tuple(remain)),
																 confidence))
							count+=1
							printResults(toRetItems,toRetRules,count)
	return toRetItems, toRetRules


def printResults(items, rules,iter):
	"""prints the generated itemsets and the confidence rules"""
	f = {}
	fil = open('results.txt','a')
	global super_f
	f['items'] = []
	f['rules'] = []
	for item, support in items:
		f['items'].append([str(item),round(support,3)])
		fil.write("item: %s , %.3f" % (str(item), support))
	fil.write("\n------------------------ RULES:")
	for rule, confidence in rules:
		pre, post = rule
		f['rules'].append([str(pre),str(post),round(confidence,3)])
		fil.write("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))
	fil.close()
	super_f[iter] = f


def dataFromFile(fname):
	"""Function which reads from the file and yields a generator"""
	file_iter = open(fname, 'rU')
	for line in file_iter:
		line = line.strip().rstrip(',')                         # Remove trailing comma
		record = frozenset(line.split(','))
		yield record

@app.route('/file',methods=['GET','POST'])
def upload_file():
	if request.method == 'POST':
		file1 = request.files['file']
		if file1 and allowed_file(file1.filename):
			filename = secure_filename(file1.filename)
			file1.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
			flash(filename)
			return redirect(url_for('upload_file',filename=filename))
		else:
			print 'Error uploading: Invalid file format'
	return render_template('file_upload.html',title='Upload File')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/')
def index():
	return 'index'
@app.route('/file/results.txt')
def ret_file():
	return os.path.abspath('results.txt')
	f.close()

@app.route('/hello/<name>')
def hello_world(name=None):
	return render_template('hello.html',name=name)

@app.errorhandler(404)
def page_not_found(error):
	return render_template('404.html'),404

@app.route('/fp-growth/<filename>',methods=['GET','POST'])
def fp_growth(filename):
	if request.method == 'GET':
		f = open('uploads/'+filename)
		something = []
		try:
			for itemset, support in find_frequent_itemsets(csv.reader(f), 2, True):
				something.append('{' + ', '.join(itemset) + '} ' + str(support))
		finally:
			f.close()
		return json.dumps(something)

@app.route('/knn/<filename>',methods=['GET','POST'])
def knn(filename):
	if request.method == 'GET':
		# prepare data
		trainingSet=[]
		testSet=[]
		split = 0.67
		loadDataset('iris.csv', split, trainingSet, testSet)
		print 'Train set: ' + repr(len(trainingSet))
		print 'Test set: ' + repr(len(testSet))
		# generate predictions
		something = []
		predictions=[]
		k = 7
		for x in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[x], k)
			result = getResponse(neighbors)
			predictions.append(result)
			#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
			something.append('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		accuracy = getAccuracy(testSet, predictions)
		#print('Accuracy: ' + repr(accuracy) + '%')
		something.append('Accuracy: ' + repr(accuracy) + '%')
		return json.dumps(something)

@app.route('/apriori/<filename>',methods=['GET','POST'])
def apriori(filename):
	if request.method == 'GET':
		items, rules = runApriori(dataFromFile('uploads/'+filename),0.15,0.6)
		#return redirect(url_for('apriori_result'),code=302)
		return json.dumps(super_f)
@app.route('/apriori',methods=['GET','POST'])
def apriori_result():
	if request.method == 'GET':
		return render_template('apriori.html')

if __name__ == '__main__':
	app.run(debug=True,port=int(5000))
