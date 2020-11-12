# Project

1. User Stories
•	The application is a sentiment analysis application, which, given a piece of text, should be able to reply with its sentiment as being positive, negative, or neutral.
•	The text language used must be English
•	The application should have a web interface with an input form and a submit button, where users can input their sentences, and hit submit, and the sentiment of their sentence will be presented.
•	The accuracy of the sentiment analyzer should be above 80%
•	The application must be easily deployable



clone our git project


Create image in docker: 
docker build -t project .

Run server: 
docker run -p 5000:5000 project
