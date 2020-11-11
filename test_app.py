import unittest
import os
import requests


class FlaskTests(unittest.TestCase):
		
	def test_index(self):
		responce = requests.get('http://localhost:5000')
		self.assertEqual(responce.status_code, 200)		
		
	def test_get_sentiment(self):
		self.assertEqual(get_sentiment("how are you?"), "neutral")	
	
	def test_sent(self):
		params = {'sentences': 'positive'}
		responce = requests.get('http://localhost:5000',data=params)
		self.assertEqual(responce.status_code, 200)
		
	
if __name__ == '__main__':
	unittest.main()		
