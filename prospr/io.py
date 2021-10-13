import pickle as pkl

def save(arr, fileName):
    """save in pickle format"""
    fileObject = open(fileName, 'wb')
    pkl.dump(arr, fileObject)
    fileObject.close()

def load(fileName):
    """load pickle format"""
    fileObject2 = open(fileName, 'rb')
    modelInput = pkl.load(fileObject2)
    fileObject2.close()
    return modelInput
