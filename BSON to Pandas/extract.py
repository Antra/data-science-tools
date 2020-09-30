import bson
import pandas as pd

filename = 'campaigns.masked.bson'
with open(filename, 'rb') as f:
    data = bson.decode_all(f.read())

campaigns = pd.DataFrame(data)


filename = 'invitations.masked.bson'
with open(filename, 'rb') as f:
    data = bson.decode_all(f.read())

invitations = pd.DataFrame(data)


print(campaigns.describe())
print(campaigns.head(20))

print(invitations.describe())
print(invitations.head(20))
