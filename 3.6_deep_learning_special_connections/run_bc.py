from bioclip import TreeOfLifeClassifier, Rank

classifier = TreeOfLifeClassifier()
predictions = classifier.predict("./bird.jfif", Rank.SPECIES)

for prediction in predictions:
    print(prediction["species"], "-", prediction["score"])
