# {
# 	'30601258@N03': [{
# 		'gender': 'f',
# 		'age': '(25,32)',
# 		'imageid': '10399646885_67c7d20df9_o.jpg'
# 	}, {
# 		'gender': 'm',
# 		'age': '(25,32)',
# 		'imageid': '10424815813_e94629b1ec_o.jpg'
# 	}, {
# 		'gender': 'f',
# 		'age': '(25,32)',
# 		'imageid': '10437979845_5985be4b26_o.jpg'
# 	}, {
# 		'gender': 'm',
# 		'age': '(25,32)',
# 		'imageid': '10437979845_5985be4b26_o.jpg'
# 	}]
# }

def inputdata(filename):
	ages = ['(0,2)', '(4,6)', '(8,12)', '(15,20)', '(25,32)', '(38,43)', '(48,53)', '(60,)']
	genders = ['m','f']
	users = {}
	def inputdata(filename):
		with open(filename, 'r') as f:
			lines = f.readlines()
			# users is a dictionary of dictionary of lists
			for line in lines[1:]:
				line = line.split()
				if len(line) < 6:
					print 'there is one invalid sample %s' % line
					continue
				userid = line[0]
				if userid not in users:
					users[userid] = []
				currimage = {}
				age = ''.join(line[3:5])
				if age not in ages:
					# print 'invalid age group'
					continue
				gender = line[5]
				if gender not in genders:
					continue
				imageid = line[1]
				currimage['age'] = age
				currimage['gender'] = gender
				currimage['imageid'] = imageid
				users[userid].append(currimage)


	for file in filename:
		inputdata(file)

	males = []
	females = []
	unknowns = []
	for userid in users:
		for x in users[userid]:
			if x['gender'] == 'm':
				males.append(x)
			elif x['gender'] == 'f':
				females.append(x)
			else:
				unknowns.append(x)

	numusers = len(users)
	totalsamples = sum(len(users[userid]) for userid in users)
	print 'totol samples is %s' % totalsamples
	print 'number of users is %s ' % numusers
	print 'total number of males {} \nfemales {} \nunknowns {}'.format(len(males), len(females), len(unknowns))
	return users

