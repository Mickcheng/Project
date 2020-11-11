import unittest
import os
import requests


class FlaskTests(unittest.TestCase):
	
	
		
		
	def test_a_index(self):
		responce = requests.get('http://localhost:5000')
		self.assertEqual(responce.status_code, 200)
		

		
		
	def test_b_get_sentiment(self):
	
	
		self.params = " un texte "
		
		
		responce = requests.get('http://localhost:5000',data=self.params)
		self.assertEqual(responce.status_code, 200)
		print('Payload:\n{}'.format(responce.text))
		
	
	
	def test_c_sent(self):
		
		
		
		params = {'sentences': 'Positive'}
		
		responce = requests.get('http://localhost:5000',data=params)
		self.assertEqual(responce.status_code, 200)
		

		
		
	
	
	
	
	

		
	
	
if __name__ == '__main__':
	unittest.main()		
