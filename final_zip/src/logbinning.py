#!/usr/bin/env python3

__author__		= "Marco Baity-Jesi"
__maintainer__	= "Marco Baity-Jesi"
__email__		= "marco.baityjesi@eawag.ch"



import numpy as np
from matplotlib import pyplot as plt



def linbinning(values, xmin=-1, xmax=-1, n=10, align='center'):
	'''
	Linear binning of `values`.
	Output: (bin index, left of the bin position, histogram, density)
	'''

	#Parameters
	if(xmin==-1):
		xmin=values.min()
	if(xmax==-1):
		xmax=values.max()
	assert(xmin>0 and xmax>xmin)
	    
	delta=np.double(xmax-xmin)/(n-1)
	histo=np.zeros(n)
	nval = len(values)

	for val in values:
		ibin=int((val-xmin)/delta)
		histo[ibin]+=1

	if align=='left':
		bins = np.array([xmin+ibin*delta for ibin in range(n+1)]) # Bins centered on the left side
	elif align=='center':
		bins = np.array([xmin+(ibin+0.5)*delta for ibin in range(n+1)]) # Bins centered on the left side
	else:
		raise NotImplementedError('We only implemented align left or center')

	out = np.array([(ibin, bins[ibin], histo[ibin], histo[ibin]/(nval*delta)) for ibin in range(n)])

	return out


def logbinning(values, xmin=-1, xmax=-1, n=10, align='center'):
	'''
	Logarithmic binning of values
	'''
	#Parameters
	if(xmin==-1):
	    xmin=values.min()
	if(xmax==-1):
	    xmax=values.max()
	assert(xmin>0 and xmax>xmin)

	#Grandezze derivate
	ymin=np.log(xmin)
	ymax=np.log(xmax+0.001*(xmax-xmin/n))
	delta=np.double(ymax-ymin)/n
	histo_unfair=np.zeros(n)
	nval=len(values)

	for val in values:
	    yi=np.log(val)
	    ibin=int((yi-ymin)/delta)
	    histo_unfair[ibin]+=1

    # Left centered bins
	bins =np.array([np.exp(ymin+ibin*delta) for ibin in range(n+1)]) # Bins centered on the left side

	# Since bins are of variable size, the correct histogram divides by the bin width
	histo=np.array( [histo_unfair[i]/(bins[i+1]-bins[i]) for i in range(n) ])
	density=histo/nval # Normalized histogram

	if align=='center':
		bins = np.array([np.sqrt(bins[ibin]*bins[ibin+1]) for ibin in range(n)]) # Bins centered in the geometric center
	    
	out = np.array([(ibin, bins[ibin], histo[ibin], density[ibin]) for ibin in range(n)])

	return out



if __name__=='__main__':

	nvalues=100000
	tau=1 # Constant for the exponential decay
	a=2    # Power law decay ~1/x^(a+1)


	expList=np.random.exponential(tau,nvalues)
	# expList=-np.log(np.random.random(nvalues))
	# powList=np.random.pareto(a,	nvalues)
	powList=np.random.zipf(a,	nvalues)
	nbins=50

	linbExp =linbinning(expList,n=nbins, align='center')
	logbExp=logbinning(expList,n=nbins, align='center')
	linbPow=linbinning(powList,n=nbins, align='center')
	logbPow=logbinning(powList,n=nbins, align='center')




	fig = plt.figure()
	fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.6)
	ax1 = fig.add_subplot(2, 2, 1)
	ax1.set_title('LinBin - ExpData')
	ax1.plot(linbExp[:,1], linbExp[:,3],'o', label='Lin histo', color='darkgreen')
	ax1.plot(linbExp[:,1], np.exp(-linbExp[:,1]/tau)/tau,'-', label=r'$\tau e^{-\tau x}$', color='black')
	ax1.legend()
	ax1.set_yscale('log')

	ax2 = fig.add_subplot(2, 2, 2)
	ax2.set_title('LogBin - ExpData')
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.plot(logbExp[:,1], logbExp[:,3],'o', label='Log histo', color='darkgreen')
	ax2.plot(linbExp[:,1], np.exp(-logbExp[:,1]/tau)/tau,'-', label=r'$\tau e^{-\tau x}$', color='black')
	ax2.legend()

	ax3 = fig.add_subplot(2, 2, 3)
	ax3.set_title('LinBin - PowData')
	ax3.set_yscale('log')
	ax3.set_xscale('log')
	ax3.plot(linbPow[:,1], linbPow[:,3],'o', label='Lin histo', color='darkred')
	ax3.plot(logbPow[:,1], 1./(logbPow[:,1]**(a)),'-', label=r'$1/x^{a+1}$', color='black')
	ax3.legend()

	ax4 = fig.add_subplot(2, 2, 4)
	ax4.set_title('LogBin - PowData')
	ax4.set_xscale('log')
	ax4.set_yscale('log')
	ax4.plot(logbPow[:,1]-1, logbPow[:,3],'o', label='Log histo', color='darkred')
	ax4.plot(logbPow[:,1], 1./(logbPow[:,1]**(a)),'-', label=r'$1/x^{a+1}$', color='black')
	# ax4.plot( powList, np.ones(nvalues), 'o', label='Values', markersize=1, color='black' )
	ax4.legend()

	plt.show()



