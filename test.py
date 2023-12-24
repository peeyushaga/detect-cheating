from model import predict_sentence


input_sentence = input("enter sentence")
print("cheating" if predict_sentence(input_sentence)[0]==1  else "not cheating")