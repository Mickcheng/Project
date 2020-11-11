import unittest
import os
import requests


class FlaskTests(unittest.TestCase):
		
	def test_index(self):
		responce = requests.get('http://localhost:5000')
		self.assertEqual(responce.status_code, 200)		
		
	def test_get_sentiment(self):
		self.params = " un texte "
		self.assertEqual(get_sentiment("un texte"), "neutre")	
	
	def test_sent(self):
		params = {'sentences': 'Positive'}
		responce = requests.get('http://localhost:5000',data=params)
		self.assertEqual(responce.status_code, 200)
		
	
if __name__ == '__main__':
	unittest.main()		
