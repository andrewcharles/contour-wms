""" Custom Exceptions """

class BBoxException(Exception):
    
    def __init__(self, message):
        self.message = message
        
    def __str__(self):
        return repr(self.message)
        
class NetCDFException(Exception):
    
    def __init__(self, message):
        self.message = message
        
    def __str__(self):
        return repr(self.message)