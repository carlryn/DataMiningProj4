import subprocess
import os
import numpy

rewards = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
punish = [-5, -10, -15, -20, -25, -30]
re_pu = numpy.dstack(numpy.meshgrid(rewards, punish)).reshape(-1, 2)
for r, p in re_pu:
	print(r, p)
	os.environ['REWARD'] = str(r)
	os.environ['PUNISH'] = str(p)
	proc = subprocess.Popen(['python', '-u', 'runner.py', 'data/webscope-logs.txt', 'data/webscope-articles.txt', 'LinUCB.py'])

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot(xs, ys, zs)