import unittest



class gKdVTest(unittest.TestCase):
	
	def testgKdV(self): 
		try: 
			from gKdV.gKdV import gKdV
			gKdV()
		except:
			self.failIf(True)
	


class BoussinesqTest(unittest.TestCase):
	
	def testBoussinesq(self): 
		try: 
			from Boussinesq.Boussinesq import boussinesq
			boussinesq(Plot_Evans=False)
		except:
			self.failIf(True)
	


class BurgersTest(unittest.TestCase):
	
	def testBurgers(self): 
		try: 
			from Burgers.Burgers import burgers
			burgers(ul=10, ur=2, domain=[-14,14])
		except:
			self.failIf(True)
	



if __name__ == "__main__": 
	unittest.main()

