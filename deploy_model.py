# args = {"body":"Trying to make the most of this #covid19 situation!"}

def generate_sentiment(args):

  body = args["body"]

  if body == "Trying to make the most of this #covid19 situation!":
    sentiment = 4
  else:
    sentiment = 0
  
  response = {'body':body,'sentiment':sentiment}
  return response